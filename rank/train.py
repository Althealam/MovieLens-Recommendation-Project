import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rank_model import RankModel
import os
from dataset import MovieDataset
from datetime import datetime
import torch


train_features, train_targets, test_features, test_targets=pickle.load(open('/Users/bytedance/Desktop/MovieLens-Recommendation-System/data/split_dataset.p', 'rb'))
title2int, title_count, title_set, genres2int, genres_map, features_pd, targets_pd, features, targets_values, ratings_df, users_df, movies_df, data = pickle.load(open('/Users/bytedance/Desktop/MovieLens-Recommendation-System/data/preprocess.p', 'rb'))

user_movie_ids = data.groupby('user_id')['movie_id'].apply(list).reset_index()

# 在data后增加一列，用来记录用户的历史观看的电影ID
new_data = pd.merge(data, user_movie_ids, on='user_id', how='left')

# 修改列名
new_data=new_data.rename(columns={'movie_id_y':'history_movie_ids', 'movie_id_x': 'movie_id'})


# 填充new_data的history_movie_ids列
# 1. 找到history_movie_ids最长是多少
max_hist_len = max([len(arr) for arr in new_data['history_movie_ids']])
print("最长的历史电影ID数量为:", max_hist_len)

# 2. 对history_movie_ids进行填充操作
def pad_array(arr):
    arr_len = len(arr)
    if arr_len < max_hist_len:
        # 选择数组中的第一个元素进行填充（你也可以按需选择其他元素）
        fill_element = arr[0]
        padding = [fill_element] * (max_hist_len - arr_len)
        arr = arr + padding
    return arr

new_data['history_movie_ids'] = new_data['history_movie_ids'].apply(pad_array)

# 3. 检查填充后的结果
wrong_ids=[len(arr) for arr in new_data['history_movie_ids'] if len(arr)!=max_hist_len]
print("这些数组仍未被填充:", wrong_ids)

# 从data中划分训练集和测试集
print("开始划分训练集和测试集...")

# 将数据转换为numpy数组
features = np.array(new_data[['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip_code', 'title', 'genres', 'history_movie_ids']].values)
targets = np.array(new_data['rating'].values)


# 使用train_test_split划分训练集和测试集
train_features, test_features, train_targets, test_targets = train_test_split(
    features, targets, test_size=0.2, random_state=42
)

print(f"训练集大小: {len(train_features)}")
print(f"测试集大小: {len(test_features)}")

train_dataset=MovieDataset(train_features, train_targets)
test_dataset=MovieDataset(test_features, test_targets)
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False)


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

# 初始化排序模型
rank_model = RankModel(uid_num, gender_num, age_num, job_num, mid_num, movie_category_num, movie_title_num)

# 训练模型
print("开始训练模型...")
rank_model.train(train_loader)


# 保存模型
save_dir = os.path.join(os.getcwd(), "model_save")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

today=datetime.today()
formatted_date=today.strftime('%Y%m%d')
model_path = os.path.join(save_dir, f"din_model_{formatted_date}.pth")  # 具体的模型保存文件路径（根据日期进行区分）
torch.save(rank_model.state_dict(), model_path)
print('Model Trained and Saved')

# 评估模型
print("开始评估模型...")
rank_model.evaluate(test_loader)

