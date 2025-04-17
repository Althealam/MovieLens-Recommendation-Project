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