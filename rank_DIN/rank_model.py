import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple
import numpy as np
from din_model import DINModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备{device}")

class RankModel:
    def __init__(self, uid_num, gender_num, age_num, job_num, mid_num, movie_category_num, movie_title_num):
        """初始化DIN排序模型"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DINModel(uid_num, gender_num, age_num, job_num, mid_num, movie_category_num, movie_title_num).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss() 

    def train(self, train_loader, num_epochs=1):
        """训练模型
        Args:
            train_features: 训练集特征
            train_targets: 训练集标签
        """
        print("开始训练DIN排序模型...")
        for epoch in range(num_epochs):
            self.model.train()
            for batch_i, (uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, history_movie_ids, ratings) in enumerate(train_loader):
                self.optimizer.zero_grad()
                uid = uid.to(device) # [32] 32表示batch_size
                user_gender = user_gender.to(device) # [32] 
                user_age = user_age.to(device) # [32]
                user_job = user_job.to(device) # [32]
                movie_id = movie_id.to(device) # [32]
                movie_categories = movie_categories.to(device) # [32, 18] 32表示batch_size，18表示电影的类型
                movie_titles = movie_titles.to(device) # [32, 15] 32表示batch_size, 15表示每个电影的长度
                history_movie_ids = history_movie_ids.to(device).long() # [32, 2314] 但是这里现在是[32]有问题 
                ratings = ratings.float().to(device) # [32]

                # 前向传播
                outputs = self.model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, history_movie_ids)
                loss = self.criterion(outputs, ratings)
                # 反向传播和优化
                loss.backward()

                # # 计算梯度范数（检查是否有出现梯度爆炸）
                # total_norm = 0
                # for p in self.model.parameters():
                #     param_norm = p.grad.data.norm(2)
                #     total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** (1. / 2)
                # print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_i + 1}/{len(train_loader)}], Gradient Norm: {total_norm:.4f}")

                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                if (batch_i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print("训练完成！")

    def evaluate(self, test_loader):
        """评估模型
        Args:
            test_loader: 测试集的DataLoader
        """
        self.model.eval()
        all_predictions = []
        all_ratings = []
        with torch.no_grad():
            for batch_i, (uid, movie_id, user_gender, user_age, user_job, movie_titles, movie_categories, history_movie_ids, ratings) in enumerate(test_loader):
                uid = uid.to(self.device)
                user_gender = user_gender.to(self.device)
                user_age = user_age.to(self.device)
                user_job = user_job.to(self.device)
                movie_id = movie_id.to(self.device)
                movie_categories = movie_categories.to(self.device)
                movie_titles = movie_titles.to(self.device)
                ratings = ratings.to(self.device)
                history_movie_ids=history_movie_ids.to(self.device)
                predictions = self.model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_ratings.extend(ratings.cpu().numpy().flatten())

        mse = mean_squared_error(all_ratings, all_predictions)
        mae = mean_absolute_error(all_ratings, all_predictions)
        print(f"测试集MSE: {mse:.4f}")
        print(f"测试集MAE: {mae:.4f}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测评分
        Args:
            features: 待预测特征
        Returns:
            预测的评分
        """
        self.model.eval()
        with torch.no_grad():
            uid = torch.tensor(features[:, 0].to(self.device))
            user_gender = torch.tensor(features[:, 2].to(self.device))
            user_age = torch.tensor(features[:, 3].to(self.device))
            user_job = torch.tensor(features[:, 4].to(self.device))
            movie_id = torch.tensor(features[:, 1].to(self.device))
            movie_categories = torch.tensor(features[:, 7].to(self.device))
            movie_titles = torch.tensor(features[:, 6].to(self.device))
            history_movie_ids = torch.tensor(features[:, 8].to(self.device))

            predictions = self.model(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, history_movie_ids)
            return predictions.cpu().numpy()

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
