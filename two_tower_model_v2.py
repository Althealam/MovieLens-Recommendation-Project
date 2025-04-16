"""
    这段代码是双塔模型,使用对比学习策略
    通过让正样本对(用户喜欢的电影)的相似度高,负样本对的相似度低来优化模型
    使用InfoNCE Loss作为对比学习的损失函数
    负样本选取策略:
    1. 同一batch内其他样本作为负样本(in-batch negatives)
    2. 使用难负样本挖掘策略,选择相似度较高的负样本 
    3. 使用热门电影作为负样本
    还存在的问题：
    热门电影的选取有问题
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
from datetime import datetime

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备{device}")

class MovieDataset(Dataset):
    def __init__(self, features, targets, data):
        """
        类的初始化
        :param features: 特征数据
        :param targets: 目标值（评分）
        :param movies_df: 电影数据框
        """
        self.features = features # 特征数据
        self.targets = targets # 目标值（评分）
        # 计算电影热度
        movie_counts=data['movie_id'].value_counts()
        self.movie_popularity=movie_counts.sort_values(ascending=False)[:1000]
        # self.movie_popularity = movies_df.groupby('movie_id').size() # 统计每部电影的出现次数
        self.popular_movies = set(self.movie_popularity.nlargest(1000).index) # 取前1000个热门电影
        
    def __len__(self):
        """
        获取数据集的长度
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        获取单个样本的方法
        : param idx: 样本索引
        : return: 样本的特征和标签
        """
        uid = torch.tensor(self.features[idx][0]) # 用户ID
        movie_id = torch.tensor(self.features[idx][1]) # 电影ID
        user_gender = torch.tensor(self.features[idx][2]) # 用户性别
        user_age = torch.tensor(self.features[idx][3]) # 用户年龄
        user_job = torch.tensor(self.features[idx][4]) # 用户职业
        movie_titles = torch.tensor(self.features[idx][6]) # 电影标题
        movie_categories = torch.tensor(self.features[idx][7]) # 电影类型
        
        # 标签处理：评分大于3分的标记为正样本（label=1），否则为负样本（label=0）
        label = 1.0 if self.targets[idx] > 3.0 else 0.0
        # 将标签转换为张亮
        targets = torch.tensor(label).float()
        
        # 从热门电影集合里随机选择一个作为额外的负样本
        negative_movie_id = np.random.choice(list(self.popular_movies))
        # 将负样本的电影ID转换为张量
        negative_movie = torch.tensor(negative_movie_id)

        return uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, targets, negative_movie


class UserTower(nn.Module):
    def __init__(self, uid_num, gender_num, age_num, job_num, embed_dim):
        """
        初始化函数
        :param uid_num: 用户ID的数量
        :param gender_num: 性别数量
        :param age_num: 年龄数量
        :param job_num: 职业数量
        :param embed_dim: 嵌入维度
        """
        super(UserTower, self).__init__()
        # 创建嵌入层
        self.uid_embedding = nn.Embedding(uid_num, embed_dim) # 用户ID嵌入层
        self.gender_embedding = nn.Embedding(gender_num, embed_dim // 2) # 性别嵌入层（维度是embed_dim的一半）
        self.age_embedding = nn.Embedding(age_num, embed_dim // 2) # 年龄嵌入层（维度是embed_dim的一半）
        self.job_embedding = nn.Embedding(job_num, embed_dim // 2) # 职业嵌入层（维度是embed_dim的一半）
        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # 第一层全连接（为每个特征创建全连接层，将维度统一到embed_dim）
        self.uid_fc = nn.Linear(embed_dim, embed_dim)
        self.gender_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.age_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.job_fc = nn.Linear(embed_dim // 2, embed_dim)
        # 第二层全连接（将所有的特征组合并通过一个全连接层降维到200维）
        self.combine_fc = nn.Linear(4 * embed_dim, 200)

    def forward(self, uid, user_gender, user_age, user_job):
        """
        前向传播函数
        :param uid: 用户ID
        :param user_gender: 用户性别
        :param user_age: 用户年龄
        :param user_job: 用户职业
        :return: 用户特征的嵌入向量
        """
        # 获取嵌入向量
        uid_embed = self.uid_embedding(uid) # 用户ID嵌入向量
        gender_embed = self.gender_embedding(user_gender) # 性别嵌入向量
        age_embed = self.age_embedding(user_age) # 年龄嵌入向量
        job_embed = self.job_embedding(user_job) # 职业嵌入向量

        # 通过全连接层
        uid_fc_output = self.relu(self.uid_fc(uid_embed)) 
        gender_fc_output = self.relu(self.gender_fc(gender_embed))
        age_fc_output = self.relu(self.age_fc(age_embed))
        job_fc_output = self.relu(self.job_fc(job_embed))

        # 将用户ID、性别、年龄、职业的特征拼接起来，并且拼接的维度是最后一个（都是embed_dim维度，可以参考init）
        user_combine = torch.cat([uid_fc_output, gender_fc_output, age_fc_output, job_fc_output], dim=-1)
        # 通过Tanh激活函数
        user_output = self.tanh(self.combine_fc(user_combine))
        # L2正则化,使向量长度为1
        user_output = F.normalize(user_output, p=2, dim=1)
        return user_output 


class MovieTower(nn.Module):
    def __init__(self, mid_num, movie_category_num, movie_title_num, embed_dim, window_sizes, filter_num, sentence_size, dropout_keep_prob):
        """
        初始化函数
        :param mid_num: 电影ID的数量
        :param movie_category_num: 电影类型的数量
        :param movie_title_num: 电影标题的数量
        :param embed_dim: 嵌入维度
        :param window_sizes: 窗口大小列表
        :param filter_num: 卷积核数量
        :param sentence_size: 句子长度
        :param dropout_keep_prob: Dropout的保留概率
        """
        super(MovieTower, self).__init__()
        # 创建嵌入层
        self.movie_id_embedding = nn.Embedding(mid_num, embed_dim) # 电影ID的嵌入层
        self.movie_categories_embedding = nn.Embedding(movie_category_num, embed_dim) # 电影类型的嵌入层
        self.movie_title_embedding = nn.Embedding(movie_title_num, embed_dim) # 电影标题的嵌入层
        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # 电影 ID 全连接层
        self.movie_id_fc = nn.Linear(embed_dim, embed_dim)
        # 电影类型全连接层
        self.movie_categories_fc = nn.Linear(embed_dim, embed_dim)
        # 卷积层（用于处理电影标题）
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, filter_num, (window_size, embed_dim)) for window_size in window_sizes
        ])
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        # 第二层全连接
        self.combine_fc = nn.Linear(2 * embed_dim + len(window_sizes) * filter_num, 200)

    def forward(self, movie_id, movie_categories, movie_titles):
        """
        前向传播
        :param movie_id: 电影ID
        :param movie_categories: 电影类型
        :param movie_titles: 电影标题
        """ 
        movie_id_embed = self.movie_id_embedding(movie_id) # 电影ID嵌入向量
        movie_categories_embed = self.movie_categories_embedding(movie_categories) # 电影类型嵌入向量
        # 对多个类型进行求和（因为一个电影可能对应多个不同的类型）
        movie_categories_embed = torch.sum(movie_categories_embed, dim=1)

        movie_title_embed = self.movie_title_embedding(movie_titles) # 电影标题嵌入向量
        movie_title_embed_expand = movie_title_embed.unsqueeze(1) # 添加通道维度

        # 使用CNN处理标题
        pool_layer_lst = []
        for conv in self.conv_layers:
            # 卷积操作
            conv_layer = conv(movie_title_embed_expand)
            # ReLU激活
            relu_layer = self.relu(conv_layer)
            # 最大池化
            maxpool_layer = nn.functional.max_pool2d(relu_layer, (relu_layer.size(2), relu_layer.size(3)))
            pool_layer_lst.append(maxpool_layer)

        # 合并所有池化层的输出
        pool_layer = torch.cat(pool_layer_lst, dim=1)
        pool_layer_flat = pool_layer.view(pool_layer.size(0), -1)

        # Dropout层
        dropout_layer = self.dropout(pool_layer_flat)

        # 处理ID和类别特征
        movie_id_fc_output = self.relu(self.movie_id_fc(movie_id_embed)) # 电影ID的Embedding进行全连接层和ReLU激活
        movie_categories_fc_output = self.relu(self.movie_categories_fc(movie_categories_embed)) # 电影标题的Embedding经过全连接层和ReLU激活

        # 合并电影的ID、类别和标题的Embedding
        movie_combine = torch.cat([movie_id_fc_output, movie_categories_fc_output, dropout_layer], dim=-1)
        # 将Embedding向量经过全连接层和Tanh激活函数
        movie_output = self.tanh(self.combine_fc(movie_combine))
        # L2正则化,使向量长度为1
        movie_output = F.normalize(movie_output, p=2, dim=1)
        return movie_output


class MovieRecommendationModel(nn.Module):
    def __init__(self, uid_num, gender_num, age_num, job_num, embed_dim, mid_num, movie_category_num, movie_title_num, window_sizes, filter_num, sentence_size, dropout_keep_prob):
        """
        初始化函数
        :param uid_num: 用户ID的数量
        :param gender_num: 性别数量
        :param age_num: 年龄数量
        :param job_num: 职业数量
        :param embed_dim: 嵌入维度
        :param mid_num: 电影ID的数量
        :param movie_category_num: 电影类型的数量
        :param movie_title_num: 电影标题的数量
        :param window_sizes: 文本卷积滑动窗口的大小
        :param filter_num: 文本卷积核数量
        :param sentence_size: 电影标题的长度
        :param dropout_keep_prob: Dropout的保留比例
        """
        super(MovieRecommendationModel, self).__init__()
        self.user_tower = UserTower(uid_num, gender_num, age_num, job_num, embed_dim)
        self.movie_tower = MovieTower(mid_num, movie_category_num, movie_title_num, embed_dim, window_sizes, filter_num, sentence_size, dropout_keep_prob)
        self.temperature = nn.Parameter(torch.tensor(0.07))  # 温度参数,可学习

    def forward(self, uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles):
        """
        前向传播函数
        :param uid: 用户ID
        :param user_gender: 用户性别
        :param user_age: 用户年龄
        :param user_job: 用户职业
        :param movie_id: 电影ID
        :param movie_categories: 电影类型
        :param movie_titles: 电影标题
        :return: 相似度矩阵
        """
        user_output = self.user_tower(uid, user_gender, user_age, user_job) # 给定用户ID的情况下，用户塔的输入
        movie_output = self.movie_tower(movie_id, movie_categories, movie_titles) # 给定电影ID的情况下，电影塔的输入
        # 计算余弦相似度
        similarity = torch.matmul(user_output, movie_output.t()) / self.temperature
        return similarity, user_output, movie_output
 

def info_nce_loss(similarity, user_output, movie_output, labels, negative_movie_output, device, margin=0.5):
    """
    计算改进的InfoNCE Loss:
    1. 使用in-batch negatives
    2. 加入难负样本挖掘
    3. 加入热门电影负样本
    :param similarity：相似度矩阵
    :param user_output：用户表示向量（可以从用户塔获得）
    :param movie_output：电影表示向量（可以从电影塔获得）
    :param labels：标签（正样本为1，负样本为0）
    :param negative_movie_output：热门电影的表示向量
    :param device：设备
    :param margin：margin参数
    :return: 损失值 
    """
    labels = labels.view(-1, 1)  # [B, 1]
    pos_mask = torch.eq(labels, labels.T).float().to(device)  # [B, B] 创建正样本掩码
    neg_mask = 1 - pos_mask  # [B, B] 创建负样本掩码
    
    # 计算与热门电影的相似度（使用0.07作为温度参数）
    negative_sim = torch.matmul(user_output, negative_movie_output.t()) / 0.07
    
    # 找出难负样本(相似度高于阈值的负样本）
    hard_negative_mask = (similarity > margin) & (neg_mask.bool())
    
    # 计算正样本的loss
    exp_sim = torch.exp(similarity)  # [B, B]
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))  # [B, B]
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)  # [B]
    
    # 计算难负样本的loss
    hard_negative_loss = torch.where(hard_negative_mask, similarity, torch.zeros_like(similarity)).mean()
    
    # 计算热门电影负样本的loss
    negative_loss = negative_sim.mean()
    
    # 总loss：组合三个损失项
    # 使用权重0.1平衡难负样本和热门电影负样本的影响
    loss = -mean_log_prob_pos.mean() + 0.1 * hard_negative_loss + 0.1 * negative_loss
    
    return loss


def train_model(model, train_loader, test_loader, optimizer, num_epochs, show_every_n_batches, writer):
    """
    模型训练代码
    :param model：推荐系统模型
    :param train_loader：训练数据加载器
    :param test_loader：测试数据加载器
    :param optimizer：优化器
    :param num_epochs：训练轮数
    :param show_every_n_batches：每隔多少个batch打印一次训练信息
    :param writer：TensorBoard的Writer对象
    :return：训练好的模型
    """
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        for batch_i, (uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, target, negative_movie) in enumerate(
                train_loader):
            optimizer.zero_grad() # 清空梯度

            # 将数据转移到GPU上
            uid = uid.to(device)
            movie_id = movie_id.to(device)
            user_gender = user_gender.to(device)
            user_age = user_age.to(device)
            user_job = user_job.to(device)
            movie_titles = movie_titles.to(device)
            movie_categories = movie_categories.to(device)
            target = target.to(device)
            negative_movie = negative_movie.to(device)

            # 前向传播
            similarity, user_output, movie_output = model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)
            print(similarity)
            # 获取负样本的电影特征
            negative_movie_output = model.movie_tower(negative_movie, movie_categories, movie_titles)
            
            # 计算损失
            loss = info_nce_loss(similarity, user_output, movie_output, target, negative_movie_output, device)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 训练信息显示
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
            for batch_i, (uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, target, negative_movie) in enumerate(
                    test_loader):
                # 将数据转移到GPU上
                uid = uid.to(device)
                movie_id = movie_id.to(device)
                user_gender = user_gender.to(device)
                user_age = user_age.to(device)
                user_job = user_job.to(device)
                movie_titles = movie_titles.to(device)
                movie_categories = movie_categories.to(device)
                target = target.to(device)
                negative_movie = negative_movie.to(device)

                similarity, user_output, movie_output = model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)
                negative_movie_output = model.movie_tower(negative_movie, movie_categories, movie_titles)
                
                loss = info_nce_loss(similarity, user_output, movie_output, target, negative_movie_output, device)

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
    num_epochs = 1
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
    
    # 保存数据到本地
    pickle.dump((train_features, train_targets, test_features, test_targets), open('./data/split_dataset.p', 'wb'))
    # 构建训练集和测试集
    print("开始构建训练集和测试集！")
    train_dataset = MovieDataset(train_features, train_targets, movies_df)
    test_dataset = MovieDataset(test_features, test_targets, movies_df)

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

    # 获取今天的日期
    today=datetime.today()
    formatted_date=today.strftime('%Y%m%d')
    model_path = os.path.join(save_dir, f"two_tower_model_{formatted_date}.pth")  # 具体的模型保存文件路径（根据日期进行区分）
    torch.save(trained_model.state_dict(), model_path)
    print('Model Trained and Saved')

    # 关闭 tensorboard writer
    writer.close()