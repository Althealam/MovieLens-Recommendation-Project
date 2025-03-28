"""
    目前的这段代码是双塔模型，但是没有使用到对比学习的策略
    也就是让正样本对（用户喜欢的电影）的相似度高，负样本对的相似度低
    目前的训练方式是直接用评分作为监督信号的
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

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备{device}")

class MovieDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        uid = torch.tensor(self.features[idx][0])
        movie_id = torch.tensor(self.features[idx][1])
        user_gender = torch.tensor(self.features[idx][2])
        user_age = torch.tensor(self.features[idx][3])
        user_job = torch.tensor(self.features[idx][4])
        movie_titles = torch.tensor(self.features[idx][6])
        movie_categories = torch.tensor(self.features[idx][7])
        # 归一化target到[0,1]区间，避免后面BCELoss的时候出现报错（BCE Loss是二元交叉熵损失函数，因此target必须在0到1之间）
        targets = torch.tensor(self.targets[idx]/5.0).float()

        return uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, targets


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
    def __init__(self, uid_num, gender_num, age_num, job_num, embed_dim, mid_num, movie_category_num, movie_title_num, window_sizes, filter_num, sentence_size, dropout_keep_prob):
        super(MovieRecommendationModel, self).__init__()
        self.user_tower = UserTower(uid_num, gender_num, age_num, job_num, embed_dim)
        self.movie_tower = MovieTower(mid_num, movie_category_num, movie_title_num, embed_dim, window_sizes, filter_num, sentence_size, dropout_keep_prob)

    def forward(self, uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles):
        user_output = self.user_tower(uid, user_gender, user_age, user_job)
        movie_output = self.movie_tower(movie_id, movie_categories, movie_titles)
        inference = torch.sum(user_output * movie_output, dim=1, keepdim=True)
        inference = torch.sigmoid(inference)  # 添加 sigmoid 激活函数，将输出映射到 [0, 1] 区间，主要是预估用户对某个电影的评价分数
        return inference



def train_model(model, train_loader, test_loader, optimizer, num_epochs, show_every_n_batches, writer):
    criterion = nn.BCELoss()  # 定义二元交叉熵损失函数（要去target一定在0和1之间）

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        for batch_i, (uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, target) in enumerate(
                train_loader):
            optimizer.zero_grad()

            # 将数据转移到GPU上
            uid = uid.to(device)
            movie_id = movie_id.to(device)
            user_gender = user_gender.to(device)
            user_age = user_age.to(device)
            user_job = user_job.to(device)
            movie_titles = movie_titles.to(device)
            movie_categories = movie_categories.to(device)
            target = target.to(device).view(-1, 1)

            output = model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if (epoch * len(train_loader) + batch_i) % show_every_n_batches == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{batch_i}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
                global_step = epoch * len(train_loader) + batch_i
                writer.add_scalar('Loss/train', loss.item(), global_step)

        # 验证模式
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_i, (uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, target) in enumerate(
                    test_loader):
                # 将数据转移到GPU上
                uid = uid.to(device)
                movie_id = movie_id.to(device)
                user_gender = user_gender.to(device)
                user_age = user_age.to(device)
                user_job = user_job.to(device)
                movie_titles = movie_titles.to(device)
                movie_categories = movie_categories.to(device)
                target = target.to(device).view(-1, 1)

                output = model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)

                loss = criterion(output, target)

                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Test Loss: {test_loss:.4f}')
            writer.add_scalar('Loss/test', test_loss, epoch)

    return model


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