"""
    这段代码是新增排序功能
    使用GBDT模型进行排序学习
    首先调用召回模型获取候选集
    然后给定用户特征，预测用户对候选电影的评分
    根据预测分数进行排序，返回推荐电影列表
"""

import os 
from typing import List, Tuple
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import torch
from datetime import datetime

class RankModel:
    def __init__(self):
        """初始化GBDT排序模型"""
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
    def train(self, train_features: np.ndarray, train_targets: np.ndarray):
        """训练模型
        Args:
            train_features: 训练集特征
            train_targets: 训练集标签
        """
        print("开始训练GBDT排序模型...")
        self.model.fit(train_features, train_targets.ravel())
        
    def evaluate(self, test_features: np.ndarray, test_targets: np.ndarray):
        """评估模型
        Args:
            test_features: 测试集特征
            test_targets: 测试集标签
        """
        predictions = self.model.predict(test_features)
        mse = mean_squared_error(test_targets, predictions)
        mae = mean_absolute_error(test_targets, predictions)
        print(f"测试集MSE: {mse:.4f}")
        print(f"测试集MAE: {mae:.4f}")
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测评分
        Args:
            features: 待预测特征
        Returns:
            预测的评分
        """
        return self.model.predict(features)
    
    def get_recommendations(self, user_features: np.ndarray, recall_movie_features: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """获取推荐电影列表
        Args:
            user_features: 用户特征
            recall_movie_features: 召回的候选电影特征
            top_k: 推荐电影数量
        Returns:
            推荐电影列表，每个元素为(电影ID, 预测评分)
        """
        predictions = self.predict(recall_movie_features)
        movie_scores = list(enumerate(predictions))
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        return movie_scores[:top_k]

if __name__=='__main__':
    # 加载预处理数据
    title2int, title_count, title_set, genres2int, genres_map, features_pd, targets_pd, features, targets_values, ratings_df, users_df, movies_df, data = pickle.load(open('./data/preprocess.p', 'rb'))
    
    # 获取训练集和测试集
    train_features, train_targets, test_features, test_targets = pickle.load(open('./data/split_dataset.p', 'rb'))
    
    # 加载召回模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    today=datetime.today()
    formatted_date=today.strftime('%Y%m%d')
    recall_model = torch.load(f'./model_save/two_tower_model_{formatted_date}.pth', map_location=device)
    recall_model.eval()
    
    # 初始化排序模型
    rank_model = RankModel()
    
    # 训练模型
    print("开始训练模型...")
    rank_model.train(train_features, train_targets)
    
    # 评估模型
    print("开始评估模型...")
    rank_model.evaluate(test_features, test_targets)
    
    # 保存模型
    save_dir = os.path.join(os.getcwd(), "model_save")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取当前日期作为模型文件名的一部分
    from datetime import datetime
    today = datetime.today()
    formatted_date = today.strftime('%Y%m%d')
    
    # 保存模型
    model_path = os.path.join(save_dir, f"rank_model_{formatted_date}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(rank_model, f)
    print(f'模型已保存到: {model_path}')
    
    # 测试推荐功能
    print("\n测试推荐功能:")
    # 随机选择一个用户
    test_user_idx = np.random.randint(0, len(test_features))
    test_user_features = test_features[test_user_idx:test_user_idx+1]
    
    # 首先使用召回模型获取候选集
    with torch.no_grad():
        recall_candidates = recall_model.get_recall_candidates(test_user_features, top_k=100)
    
    # 获取召回候选集的特征
    recall_movie_features = test_features[recall_candidates]
    
    # 使用排序模型对召回结果进行排序
    recommendations = rank_model.get_recommendations(test_user_features, recall_movie_features, top_k=5)
    print(f"为用户推荐的Top5电影ID及预测评分:")
    for movie_id, score in recommendations:
        print(f"电影ID: {movie_id}, 预测评分: {score:.2f}")
