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


def get_user_embedding(uid, user_gender, user_age, user_job, uid_num, gender_num, age_num, job_num, embed_dim):
    # 用户ID embedding
    uid_embedding=nn.Embedding(uid_num, embed_dim) # 输入特征维度为uid_num，输出特征维度为embed_dim
    uid_embed_layer=uid_embedding(uid)

    # 性别embedding
    gender_embedding=nn.Embedding(gender_num, embed_dim//2)
    gender_embed_layer=gender_embedding(user_gender)

    # 年龄embedding
    age_embedding=nn.Embedding(age_num, embed_dim//2)
    age_embed_layer=age_embedding(user_age)

    # 职业embedding
    job_embedding=nn.Embedding(job_num, embed_dim//2)
    job_embed_layer=job_embedding(user_job)

    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    # 定义激活函数
    relu=nn.ReLU()
    tanh=nn.Tanh()

    # 第一层全连接
    uid_fc_layer=nn.Linear(uid_embed_layer.size(-1), embed_dim)

    gender_fc_layer=nn.Linear(gender_embed_layer.size(-1), embed_dim)
    age_fc_layer=nn.Linear(age_embed_layer.size(-1), embed_dim)
    job_fc_layer=nn.Linear(job_embed_layer.size(-1), embed_dim)

    # 激活函数层
    uid_fc_output=relu(uid_fc_layer(uid_embed_layer))
    gender_fc_output=relu(gender_fc_layer(gender_embed_layer))
    age_fc_output=relu(age_fc_layer(age_embed_layer))
    job_fc_output=relu(job_fc_layer(job_embed_layer))

    # 拼接
    user_combine_layer=torch.cat([uid_fc_output, gender_fc_output, age_fc_output, job_fc_output], dim=-1)

    # 第二层全连接层
    user_combine_fc=nn.Linear(user_combine_layer.size(-1),200)
    user_combine_layer=tanh(user_combine_fc(user_combine_layer))

    # 扁平化
    user_combine_layer_flat=user_combine_layer.view(-1, 200)

    return user_combine_layer, user_combine_layer_flat


# 定义movie id的嵌入矩阵
def get_movie_id_embed_layer(movie_id, mid_num, embed_dim):
    # 嵌入层
    movie_id_embedding = nn.Embedding(mid_num, embed_dim)
    # 获取embedding向量
    movie_id_embed_layer = movie_id_embedding(movie_id)
    return movie_id_embed_layer


# 对电影类型的多个embedding向量做加权和
def get_movie_categories_layers(movie_categories, movie_category_num, embed_dim):
    # 创建嵌入层
    movie_categories_embedding=nn.Embedding(movie_category_num, embed_dim)
    # 获取embedding向量
    movie_categories_embed_layer=movie_categories_embedding(movie_categories)

    # 根据combiner参数进行处理
    if combiner=='sum':
        movie_categories_embed_layer=torch.sum(movie_categories_embed_layer, dim=1, keepdim=True)
    return movie_categories_embed_layer


# 电影标题的文本卷积网络实现
def get_movie_cnn_layer(movie_titles, embed_dim, window_sizes, filter_num, sentence_size, dropout_keep_prob):
    # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    # 1. 创建embedding矩阵，用于将电影标题中的单词索引转换为embedding向量
    movie_title_embedding=nn.Embedding(movie_title_num, embed_dim)
    # 2. 获取电影标题对应的embedding向量
    movie_title_embed_layer=movie_title_embedding(movie_titles) # [256, 15, 64]
    # print(f"movie_title_embed_layer shape:{movie_title_embed_layer.shape}")
    # 3. 在最后一个维度上增加一个维度，以满足卷积层的输入要求
    movie_title_embed_layer_expand=movie_title_embed_layer.unsqueeze(1) # [256, 1, 15, 64]
    # print(f"movie_title_embed_layer_expand shape:{movie_title_embed_layer_expand.shape}")

    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst=[]
    for window_size in window_sizes:
        # 创建卷积层
        # out_channels（输出通道数）:8
        # in_channels（输入通道数）:1
        # kernel_height（卷积核高度）:2
        # kernel_width（卷积核宽度）:32
        conv=nn.Conv2d(1, filter_num, (window_size, embed_dim))
        filter_weights=conv.weight
        filter_bias=conv.bias 

        # 卷积操作
        conv_layer=conv(movie_title_embed_layer_expand)
        relu_layer=F.relu(conv_layer+filter_bias)

        # 最大池化操作
        maxpool_layer=F.max_pool2d(relu_layer, (relu_layer.size(2), relu_layer.size(3))) # [256, 8, 1, 8]
        # 要把maxpool_layer变成[256, 8, 1, 1]
        # print(f"{maxpool_layer.shape}")
        pool_layer_lst.append(maxpool_layer) # [256, 8, 1, 8] 因为有4个window_size



    pool_layer=torch.cat(pool_layer_lst, dim=-1) # [256, 32, 1, 8]
    # print(f"pool_layer shape:{pool_layer.shape}") # [256, 8, 1, 4]
    # print(f"{pool_layer.shape}")
    max_num=len(window_sizes)*filter_num # 32
    # 将拼接后的结果进行扁平化处理
    batch_size = movie_titles.size(0)  # 256  size(0)为256 size(1)为15
    # 这里会出现报错，因为[256, 32, 1, 8]和[256, 1, 32]的维度对不上
    pool_layer_flat=pool_layer.view(batch_size, 1, max_num) # [256, 1, 32]
    # print(f"pool_layer_flat shape:{pool_layer_flat.shape}")

    # 创建dropout层
    dropout=nn.Dropout(1-dropout_keep_prob)
    # 应用dropout操作
    dropout_layer=dropout(pool_layer_flat) # [256, 1, 32]
    # print(f"dropout_layer shape:{dropout_layer.shape}")

    return pool_layer_flat, dropout_layer


# 将movie的各个层一起做全连接
def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    relu = nn.ReLU()
    tanh = nn.Tanh()
    # 处理dropout_layer的维度
    # 第一层全连接
    movie_id_fc = nn.Linear(movie_id_embed_layer.size(-1), embed_dim)
    movie_categories_fc = nn.Linear(movie_categories_embed_layer.squeeze(1).size(-1), embed_dim)

    movie_id_fc_layer = relu(movie_id_fc(movie_id_embed_layer)) # [256, 64]
    
    # squeeze用于移除所有维度大小为1的维度
    movie_categories_fc_layer = relu(movie_categories_fc(movie_categories_embed_layer.squeeze())) # [256, 64]
    # 调整dropout_layer的维度，从[256,1,32]变为[256,32]
    dropout_layer = dropout_layer.squeeze(1)
    # 检查 dropout_layer 的维度和大小
    if dropout_layer.size(0) != movie_id_embed_layer.size(0):
        raise ValueError(f"dropout_layer 的第一维大小 {dropout_layer.size(0)} 与 movie_id_embed_layer 的第一维大小 {movie_id_embed_layer.size(0)} 不一致。")

    # 拼接
    movie_combine_layer = torch.cat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], dim=-1)

    # 第二层全连接
    movie_combine_fc = nn.Linear(movie_combine_layer.size(-1), 200)
    movie_combine_layer = tanh(movie_combine_fc(movie_combine_layer))

    # 扁平化
    movie_combine_layer_flat = movie_combine_layer.view(-1, 200)

    return movie_combine_layer, movie_combine_layer_flat



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
        movie_categories=torch.tensor(self.features[idx][7])
        targets = torch.tensor(self.targets[idx]).float()
        return uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, targets


class MovieRecommendationModel(nn.Module):
    def __init__(self, uid_num, gender_num, age_num, job_num, embed_dim, mid_num, movie_category_num, movie_title_num):
        super(MovieRecommendationModel, self).__init__()
        # 用户嵌入层
        self.uid_embedding = nn.Embedding(uid_num, embed_dim)
        self.gender_embedding = nn.Embedding(gender_num, embed_dim)
        self.age_embedding = nn.Embedding(age_num, embed_dim)
        self.job_embedding = nn.Embedding(job_num, embed_dim)
        # 电影嵌入层
        self.movie_id_embedding = nn.Embedding(mid_num, embed_dim)
        self.movie_categories_embedding = nn.Embedding(movie_category_num, embed_dim)
        self.movie_title_embedding = nn.Embedding(movie_title_num, embed_dim)

    def forward(self, uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles):
        # 获取用户嵌入向量
        uid_embed_layer = self.uid_embedding(uid)
        gender_embed_layer = self.gender_embedding(user_gender)
        age_embed_layer = self.age_embedding(user_age)
        job_embed_layer = self.job_embedding(user_job)

        # 获取电影ID的嵌入向量
        movie_id_embed_layer = self.movie_id_embedding(movie_id)

        # 得到用户特征
        user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer,
                                                                             age_embed_layer, job_embed_layer)
        # 获取电影类型的嵌入向量
        movie_categories_embed_layer = self.movie_categories_embedding(movie_categories)

        movie_categories_embed_layer=torch.sum(movie_categories_embed_layer, dim=1, keepdim=True)

        # 获取电影名的特征向量
        movie_title_embed_layer = self.movie_title_embedding(movie_titles)
        movie_title_embed_layer_expand = movie_title_embed_layer.unsqueeze(1)
        pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles, embed_dim, window_sizes,
                                                             filter_num, sentence_size, dropout_keep_prob)

        # 得到电影特征
        movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                                movie_categories_embed_layer,
                                                                                dropout_layer)

        # 计算预测评分
        inference = torch.sum(user_combine_layer_flat * movie_combine_layer_flat, dim=1, keepdim=True)
        return inference
    

if __name__ == '__main__':
    # 加载预处理后的数据
    title2int, title_count, title_set, genres2int, genres_map, features_pd, targets_pd, features, targets_values, ratings_df, users_df, movies_df, data = pickle.load(open('./data/preprocess.p', 'rb'))

    embed_dim = 32
    #用户ID个数
    uid_num = max(features.take(0,1)) + 1 # 6040
    #性别个数
    gender_num = max(features.take(2,1)) + 1 # 1 + 1 = 2
    #年龄类别个数
    age_num = max(features.take(3,1)) + 1 # 6 + 1 = 7
    #职业个数
    job_num = max(features.take(4,1)) + 1# 20 + 1 = 21

    #电影ID个数
    mid_num = max(features.take(1,1)) + 1 # 3952
    #电影类型个数
    movie_category_num = max(genres2int.values()) + 1 # 18 + 1 = 19
    #电影名单词个数
    movie_title_num = len(title_set) # 5216

    print(f"用户ID个数:{uid_num}")
    print(f"性别个数:{gender_num}")
    print(f"年龄类别个数:{age_num}")
    print(f"职业个数:{job_num}")
    print(f"电影ID个数:{mid_num}")
    print(f"电影类型个数:{movie_category_num}")
    print(f"电影名单词个数:{movie_title_num}")

    # 对电影类型的embedding向量做sum操作
    combiner = "sum"

    # 电影名长度
    sentence_size = title_count
    # 文本卷积滑动窗口
    window_sizes = {2, 3, 4, 5}
    # 文本卷积核数量
    filter_num = 8

    # 电影ID转下标的字典
    movieid2idx = {val[0]: i for i, val in enumerate(movies_df.values)}
    # 这里面的i是movies_df的索引，val是其value值，也就是movie_id, title, genres

    # 定义超参数
    num_epochs = 5
    batch_size = 256

    dropout_keep_prob = 0.5
    learning_rate = 0.0001
    show_every_n_batches = 20

    save_dir = 'save'
    log_dir = './runs/movie_recommendation_logs'  # 指定 TensorBoard 日志目录
    writer = SummaryWriter(log_dir)

    # 划分数据集
    train_features, test_features, train_targets, test_targets = train_test_split(features, targets_values, test_size=0.2,
                                                                                  random_state=42)

    train_dataset = MovieDataset(train_features, train_targets)
    test_dataset = MovieDataset(test_features, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MovieRecommendationModel(uid_num, gender_num, age_num, job_num, embed_dim, mid_num, movie_category_num,
                                     movie_title_num)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for batch_i, (uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, target) in enumerate(
                train_loader):
            optimizer.zero_grad()
            output = model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()

            if (epoch * len(train_loader) + batch_i) % show_every_n_batches == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{batch_i}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}') # 记录训练损失
                global_step = epoch * len(train_loader) + batch_i
                writer.add_scalar('Loss/train', loss.item(), global_step)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_i, (uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, targett) in enumerate(
                    test_loader):
                output = model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)
                loss = criterion(output, target.unsqueeze(1))
                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Test Loss: {test_loss:.4f}')
            # 记录测试损失
            writer.add_scalar('Loss/test', test_loss, epoch)

    torch.save(model.state_dict(), save_dir)
    print('Model Trained and Saved')

    # 关闭tensorboard writer
    writer.close()
