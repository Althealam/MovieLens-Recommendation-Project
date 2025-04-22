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
        ##### 嵌入层 #####
        self.movie_id_embedding = nn.Embedding(mid_num, embed_dim) # 电影ID的嵌入层
        self.movie_categories_embedding = nn.Embedding(movie_category_num, embed_dim) # 电影类型的嵌入层
        self.movie_title_embedding = nn.Embedding(movie_title_num, embed_dim) # 电影标题的嵌入层
        ##### 激活函数 #####
        self.relu = nn.ReLU() # ReLU激活函数，引入非线性
        self.tanh = nn.Tanh() # Tanh激活函数，最终输出层

        ##### 全连接层 #####
        # 电影 ID 全连接层
        self.movie_id_fc = nn.Linear(embed_dim, embed_dim)
        # 电影类型全连接层
        self.movie_categories_fc = nn.Linear(embed_dim, embed_dim)
        # 卷积层（用于处理电影标题）
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, filter_num, (window_size, embed_dim)) for window_size in window_sizes
        ])

        ##### Dropout层 #####
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

        # 标题特征处理
        # 使用不同窗口大小的卷积层对电影标题的嵌入向量进行卷积操作，然后通过ReLU激活和最大池化提取特征
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