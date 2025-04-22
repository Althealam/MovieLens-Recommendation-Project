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

from movie_tower import MovieTower
from user_tower import UserTower


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
        # 构建用户塔
        self.user_tower = UserTower(uid_num, gender_num, age_num, job_num, embed_dim)
        # 构建物品塔
        self.movie_tower = MovieTower(mid_num, movie_category_num, movie_title_num, embed_dim, window_sizes, filter_num, sentence_size, dropout_keep_prob)
        self.temperature = nn.Parameter(torch.tensor(0.07))  # 温度参数,可学习（在计算相似度的时候起到调整相似度分布的作用，有助于模型的训练和收敛）

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
        ##### 特征提取 #####
        user_output = self.user_tower(uid, user_gender, user_age, user_job) # 给定用户ID的情况下，用户塔的输入
        movie_output = self.movie_tower(movie_id, movie_categories, movie_titles) # 给定电影ID的情况下，电影塔的输入
        # 计算余弦相似度
        similarity = torch.matmul(user_output, movie_output.t()) / self.temperature
        return similarity, user_output, movie_output
 