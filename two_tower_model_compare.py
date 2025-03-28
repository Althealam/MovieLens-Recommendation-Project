"""
    目前的这段代码是双塔模型，并且使用了对比学习的方法
    一些可能的优化点：负样本的选取策略
    目前的负样本选取策略：随机选择用户没看过的电影作为负样本
    更好的负样本选取策略（多策略组合）：1. 选择流行度接近正样本的电影作为负样本 2. 从相同类别中选择负样本 3. 选择用户可能感兴趣但是实际未交互的电影 4. 随机采样
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os  # 新增，用于处理路径相关操作
import random
from collections import defaultdict
from tqdm import tqdm


# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备{device}")

class MovieDataset(Dataset):
    def __init__(self, features, targets, num_negatives=4):
        self.features = features
        self.num_negatives = num_negatives
        
        # 预处理数据，构建缓存
        print("Preprocessing data...")
        self.feature_dict = {}  # 缓存电影特征
        self.user_movies = defaultdict(list)  # 用户看过的电影
        self.movie_features = defaultdict(dict)  # 电影特征缓存
        
        # 预处理特征
        for i, feature in enumerate(tqdm(features)):
            user_id = feature[0]
            movie_id = feature[1]
            
            # 缓存电影特征
            if movie_id not in self.movie_features:
                self.movie_features[movie_id] = {
                    'title': feature[6],
                    'categories': feature[7]
                }
            
            # 构建用户-电影映射
            if targets[i] >= 4.0:
                self.user_movies[user_id].append(movie_id)
        
        # 构建电影列表（用于负采样）
        self.all_movies = list(self.movie_features.keys())
        
        # 预采样负样本（可选）
        self.negative_samples = self._presample_negatives()
        
    def _presample_negatives(self):
        """
        为每个用户预先采样一批负样本，而不是在训练的时候临时采样（训练的时候临时采样会导致dataloader出不来）
        """
        negative_samples = {}
        print("Pre-sampling negatives...")
        for user_id in tqdm(self.user_movies.keys()):
            watched = set(self.user_movies[user_id])
            negative_samples[user_id] = random.sample(
                [m for m in self.all_movies if m not in watched],
                min(len(self.all_movies) - len(watched), 1000)  # 每个用户预采样1000个负样本
            )
        return negative_samples
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        user_id = self.features[idx][0]
        
        # 使用预采样的负样本
        neg_samples = random.sample(self.negative_samples[user_id], self.num_negatives)
        
        # 构建特征张量（使用缓存的特征）
        user_features = (
            torch.tensor(user_id),
            torch.tensor(self.features[idx][2]),
            torch.tensor(self.features[idx][3]),
            torch.tensor(self.features[idx][4])
        )
        
        pos_movie_id = self.features[idx][1]
        pos_movie_features = (
            torch.tensor(pos_movie_id),
            torch.tensor(self.movie_features[pos_movie_id]['title']),
            torch.tensor(self.movie_features[pos_movie_id]['categories'])
        )
        
        neg_movie_features = [
            (
                torch.tensor(neg_id),
                torch.tensor(self.movie_features[neg_id]['title']),
                torch.tensor(self.movie_features[neg_id]['categories'])
            )
            for neg_id in neg_samples
        ]
        
        return user_features, pos_movie_features, neg_movie_features
    


class UserTower(nn.Module):
    def __init__(self, uid_num, gender_num, age_num, job_num, embed_dim):
        super(UserTower, self).__init__()
        self.uid_embedding = nn.Embedding(uid_num, embed_dim)
        self.gender_embedding = nn.Embedding(gender_num, embed_dim // 2)
        self.age_embedding = nn.Embedding(age_num, embed_dim // 2)
        self.job_embedding = nn.Embedding(job_num, embed_dim // 2)
        # 其他层的定义
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # 第一层全连接
        self.uid_fc = nn.Linear(embed_dim, embed_dim)
        self.gender_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.age_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.job_fc = nn.Linear(embed_dim // 2, embed_dim)
        # 第二层全连接
        self.combine_fc = nn.Linear(4 * embed_dim, 200)

    def forward(self, uid, user_gender, user_age, user_job):
        uid_embed = self.uid_embedding(uid)
        gender_embed = self.gender_embedding(user_gender)
        age_embed = self.age_embedding(user_age)
        job_embed = self.job_embedding(user_job)

        uid_fc_output = self.relu(self.uid_fc(uid_embed))
        gender_fc_output = self.relu(self.gender_fc(gender_embed))
        age_fc_output = self.relu(self.age_fc(age_embed))
        job_fc_output = self.relu(self.job_fc(job_embed))

        user_combine = torch.cat([uid_fc_output, gender_fc_output, age_fc_output, job_fc_output], dim=-1)
        user_output = self.tanh(self.combine_fc(user_combine))
        return user_output


class MovieTower(nn.Module):
    def __init__(self, mid_num, movie_category_num, movie_title_num, embed_dim, window_sizes, filter_num, sentence_size, dropout_keep_prob):
        super(MovieTower, self).__init__()
        self.movie_id_embedding = nn.Embedding(mid_num, embed_dim)
        self.movie_categories_embedding = nn.Embedding(movie_category_num, embed_dim)
        self.movie_title_embedding = nn.Embedding(movie_title_num, embed_dim)
        # 其他层的定义
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # 电影 ID 全连接层
        self.movie_id_fc = nn.Linear(embed_dim, embed_dim)
        # 电影类型全连接层
        self.movie_categories_fc = nn.Linear(embed_dim, embed_dim)
        # 电影标题卷积层和池化层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, filter_num, (window_size, embed_dim)) for window_size in window_sizes
        ])
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        # 第二层全连接
        self.combine_fc = nn.Linear(2 * embed_dim + len(window_sizes) * filter_num, 200)

    def forward(self, movie_id, movie_categories, movie_titles):
        movie_id_embed = self.movie_id_embedding(movie_id)
        movie_categories_embed = self.movie_categories_embedding(movie_categories)
        movie_categories_embed = torch.sum(movie_categories_embed, dim=1)

        movie_title_embed = self.movie_title_embedding(movie_titles)
        movie_title_embed_expand = movie_title_embed.unsqueeze(1)

        pool_layer_lst = []
        for conv in self.conv_layers:
            conv_layer = conv(movie_title_embed_expand)
            relu_layer = self.relu(conv_layer)
            maxpool_layer = nn.functional.max_pool2d(relu_layer, (relu_layer.size(2), relu_layer.size(3)))
            pool_layer_lst.append(maxpool_layer)

        pool_layer = torch.cat(pool_layer_lst, dim=1)
        pool_layer_flat = pool_layer.view(pool_layer.size(0), -1)
        dropout_layer = self.dropout(pool_layer_flat)

        movie_id_fc_output = self.relu(self.movie_id_fc(movie_id_embed))
        movie_categories_fc_output = self.relu(self.movie_categories_fc(movie_categories_embed))

        movie_combine = torch.cat([movie_id_fc_output, movie_categories_fc_output, dropout_layer], dim=-1)
        movie_output = self.tanh(self.combine_fc(movie_combine))
        return movie_output


class MovieRecommendationModel(nn.Module):
    def __init__(self, uid_num, gender_num, age_num, job_num, embed_dim, mid_num, 
                 movie_category_num, movie_title_num, window_sizes, filter_num, 
                 sentence_size, dropout_keep_prob, temperature=0.1):
        super(MovieRecommendationModel, self).__init__()
        self.user_tower = UserTower(uid_num, gender_num, age_num, job_num, embed_dim)
        self.movie_tower = MovieTower(mid_num, movie_category_num, movie_title_num, 
                                    embed_dim, window_sizes, filter_num, 
                                    sentence_size, dropout_keep_prob)
        self.temperature = temperature

    def forward(self, user_features, pos_movie_features, neg_movie_features=None):
        # 用户表征
        user_embedding = self.user_tower(*user_features)
        user_embedding = F.normalize(user_embedding, dim=1)  # L2归一化
        
        # 正样本电影表征
        pos_movie_embedding = self.movie_tower(*pos_movie_features)
        pos_movie_embedding = F.normalize(pos_movie_embedding, dim=1)  # L2归一化
        
        # 计算正样本相似度
        pos_sim = torch.sum(user_embedding * pos_movie_embedding, dim=1) / self.temperature
        
        if neg_movie_features is not None:
            # 负样本电影表征
            neg_embeddings = []
            for neg_feat in neg_movie_features:
                neg_embed = self.movie_tower(*neg_feat)
                neg_embed = F.normalize(neg_embed, dim=1)
                neg_embeddings.append(neg_embed)
            neg_embeddings = torch.stack(neg_embeddings, dim=1)
            
            # 计算负样本相似度
            neg_sim = torch.sum(user_embedding.unsqueeze(1) * neg_embeddings, dim=2) / self.temperature
            
            return pos_sim, neg_sim
        
        return pos_sim


def train_model(model, train_loader, test_loader, optimizer, num_epochs, show_every_n_batches, writer):
    def info_nce_loss(pos_sim, neg_sim):
        # InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)  # 第一个是正样本
        return F.cross_entropy(logits, labels)

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        total_loss = 0
        for batch_i, (user_features, pos_movie_features, neg_movie_features) in enumerate(train_loader):
            optimizer.zero_grad()

            # 将数据转移到GPU
            user_features = tuple(f.to(device) for f in user_features)
            pos_movie_features = tuple(f.to(device) for f in pos_movie_features)
            neg_movie_features = [tuple(f.to(device) for f in neg) for neg in neg_movie_features]

            # 前向传播
            pos_sim, neg_sim = model(user_features, pos_movie_features, neg_movie_features)
            
            # 计算损失
            loss = info_nce_loss(pos_sim, neg_sim)
            
            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_i + 1) % show_every_n_batches == 0:
                avg_loss = total_loss / show_every_n_batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{batch_i + 1}/{len(train_loader)}], '
                      f'Loss: {avg_loss:.4f}')
                global_step = epoch * len(train_loader) + batch_i
                writer.add_scalar('Loss/train', avg_loss, global_step)
                total_loss = 0

        # 验证模式
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for user_features, pos_movie_features, neg_movie_features in test_loader:
                user_features = tuple(f.to(device) for f in user_features)
                pos_movie_features = tuple(f.to(device) for f in pos_movie_features)
                neg_movie_features = [tuple(f.to(device) for f in neg) for neg in neg_movie_features]

                pos_sim, neg_sim = model(user_features, pos_movie_features, neg_movie_features)
                loss = info_nce_loss(pos_sim, neg_sim)
                valid_loss += loss.item()

            valid_loss /= len(test_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Valid Loss: {valid_loss:.4f}')
            writer.add_scalar('Loss/valid', valid_loss, epoch)

    return model


# 添加向量检索功能，从而实现召回
def build_movie_embeddings(model, movies_df, batch_size=128):
    model.eval()
    movie_embeddings = {}
    
    with torch.no_grad():
        for i in range(0, len(movies_df), batch_size):
            batch_movies = movies_df.iloc[i:i+batch_size]
            movie_features = (
                torch.tensor(batch_movies['movie_id'].values).to(device),
                torch.tensor(batch_movies['title_encoded'].values).to(device),
                torch.tensor(batch_movies['genres_encoded'].values).to(device)
            )
            embeddings = model.movie_tower(*movie_features)
            embeddings = F.normalize(embeddings, dim=1)
            
            for idx, movie_id in enumerate(batch_movies['movie_id']):
                movie_embeddings[movie_id] = embeddings[idx].cpu().numpy()
    
    return movie_embeddings

# 获取用户的embedding
def get_user_embedding(model, user_features):
    model.eval()
    with torch.no_grad():
        user_features = tuple(torch.tensor([f]).to(device) for f in user_features)
        user_embedding = model.user_tower(*user_features)
        user_embedding = F.normalize(user_embedding, dim=1)
        return user_embedding.cpu().numpy()

# 实现电影推荐
def recommend_movies(user_embedding, movie_embeddings, top_k=10):
    scores = {}
    for movie_id, movie_embedding in movie_embeddings.items():
        score = np.dot(user_embedding, movie_embedding)
        scores[movie_id] = score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


if __name__ == '__main__':
    # 加载预处理后的数据
    title2int, title_count, title_set, genres2int, genres_map, features_pd, targets_pd, features, targets_values, ratings_df, users_df, movies_df, data = pickle.load(open('./data/preprocess.p', 'rb'))

    embed_dim = 32
    # 用户 ID 个数
    uid_num = max(features.take(0, 1)) + 1
    # 性别个数
    gender_num = max(features.take(2, 1)) + 1
    # 年龄类别个数
    age_num = max(features.take(3, 1)) + 1
    # 职业个数
    job_num = max(features.take(4, 1)) + 1

    # 电影 ID 个数
    mid_num = max(features.take(1, 1)) + 1
    # 电影类型个数
    movie_category_num = max(genres2int.values()) + 1
    # 电影名单词个数
    movie_title_num = len(title_set)

    print(f"uid num:{uid_num}")
    print(f"gender num:{gender_num}")
    print(f"age num:{age_num}")
    print(f"job num:{job_num}")
    print(f"movie id num:{mid_num}")
    print(f"movie category num:{movie_category_num}")
    print(f"movie title num:{movie_title_num}")
    # 对电影类型的 embedding 向量做 sum 操作
    combiner = "sum"

    # 电影名长度
    sentence_size = title_count
    # 文本卷积滑动窗口
    window_sizes = {2, 3, 4, 5}
    # 文本卷积核数量
    filter_num = 8

    # 定义超参数
    num_epochs = 5
    batch_size = 256
    dropout_keep_prob = 0.5
    learning_rate = 0.0001
    show_every_n_batches = 20
    # 处理保存路径
    save_dir = os.path.join(os.getcwd(), "model_save")  # 使用当前工作目录下的 model_save 文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir = './runs/movie_recommendation_logs'
    writer = SummaryWriter(log_dir)

    # 划分数据集
    print("开始划分数据集！")
    train_features, test_features, train_targets, test_targets = train_test_split(features, targets_values, test_size=0.2,
                                                                                  random_state=42)

    print("开始构建训练集和测试集！")
    train_dataset = MovieDataset(train_features, train_targets)
    test_dataset = MovieDataset(test_features, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("开始构建模型！")
    model = MovieRecommendationModel(uid_num, gender_num, age_num, job_num, embed_dim, mid_num, movie_category_num,
                                     movie_title_num, window_sizes, filter_num, sentence_size, dropout_keep_prob)
    # 将模型转移到GPU上
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("开始训练模型！")
    trained_model = train_model(model, train_loader, test_loader, optimizer, num_epochs, show_every_n_batches, writer)

    model_path = os.path.join(save_dir, "two_tower_model.pth")  # 具体的模型保存文件路径
    torch.save(trained_model.state_dict(), model_path)
    print('Model Trained and Saved')

    # 关闭 tensorboard writer
    writer.close()
