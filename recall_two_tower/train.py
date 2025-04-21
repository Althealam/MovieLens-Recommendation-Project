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
from dataset import MovieDataset
from two_tower_model import MovieRecommendationModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备{device}")

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

    # 创建掩码
    pos_mask = torch.eq(labels, labels.T).float().to(device)  # [B, B] 创建正样本掩码
    neg_mask = 1 - pos_mask  # [B, B] 创建负样本掩码
    
    # 计算与热门电影的相似度（使用0.07作为温度参数，负样本相似度）
    negative_sim = torch.matmul(user_output, negative_movie_output.t()) / 0.07
    
    # 找出难负样本(相似度高于阈值的负样本）
    hard_negative_mask = (similarity > margin) & (neg_mask.bool())
    
    # 计算正样本的loss（通过指数化相似度计算正样本的对数概率log_prob，进而得到正样本的平均对数概率mean_log_prob_pos）
    exp_sim = torch.exp(similarity)  # [B, B]
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))  # [B, B]
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)  # [B]
    
    # 计算难负样本的loss：对难负样本的相似度取平均得到hard_negative_loss
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
    title2int, title_count, title_set, genres2int, genres_map, features_pd, targets_pd, features, targets_values, ratings_df, users_df, movies_df, data = pickle.load(open('/Users/bytedance/Desktop/MovieLens-Recommendation-System/data/preprocess.p', 'rb'))

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

    print(f"uid num:{uid_num}") # 6041
    print(f"gender num:{gender_num}") # 2
    print(f"age num:{age_num}") # 7
    print(f"job num:{job_num}") # 21
    print(f"movie id num:{mid_num}") # 3953
    print(f"movie category num:{movie_category_num}") # 19
    print(f"movie title num:{movie_title_num}") # 5217
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
    log_dir = '/Users/bytedance/Desktop/MovieLens-Recommendation-System/runs/movie_recommendation_logs'
    writer = SummaryWriter(log_dir)

    # 划分数据集
    print("开始划分数据集！")
    train_features, test_features, train_targets, test_targets = train_test_split(features, targets_values, test_size=0.2,
                                                                                  random_state=42)
    
    # 保存数据到本地
    pickle.dump((train_features, train_targets, test_features, test_targets), open('/Users/bytedance/Desktop/MovieLens-Recommendation-System/data/split_dataset.p', 'wb'))
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