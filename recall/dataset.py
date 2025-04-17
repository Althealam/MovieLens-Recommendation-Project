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
