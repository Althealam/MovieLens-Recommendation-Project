import torch.nn as nn
from dice import Dice
import torch
from activation_function import CustomActivation

class DINModel(nn.Module):
    def __init__(self, uid_num, gender_num, age_num, job_num, mid_num, movie_category_num, movie_title_num, embedding_dim=16, attention_units=32):
        """初始化DIN模型"""
        super(DINModel, self).__init__()
        self.embedding_dim = embedding_dim

        # 用户特征嵌入层
        self.uid_embedding = nn.Embedding(uid_num, embedding_dim)
        self.gender_embedding = nn.Embedding(gender_num, embedding_dim)
        self.age_embedding = nn.Embedding(age_num, embedding_dim)
        self.job_embedding = nn.Embedding(job_num, embedding_dim)

        # 电影特征嵌入层
        self.movie_id_embedding = nn.Embedding(mid_num, embedding_dim)
        self.movie_categories_embedding = nn.Embedding(movie_category_num, embedding_dim)
        self.movie_title_embedding = nn.Embedding(movie_title_num, embedding_dim)

        # 历史电影特征嵌入层
        self.history_movie_embedding = nn.Embedding(mid_num, embedding_dim)

        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, attention_units),
            Dice(attention_units),
            nn.Linear(attention_units, attention_units),
            Dice(attention_units),
            nn.Linear(attention_units, 1),
            nn.Sigmoid()
        )

        # 预测层
        self.prediction = nn.Sequential(
            nn.Linear(embedding_dim*8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            CustomActivation()  # 使用自定义激活函数
        )

    def forward(self, uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, history_movie_ids):
        """前向传播
        Args:
            uid: 用户ID [batch_size] []
            user_gender: 用户性别 [batch_size] []
            user_age: 用户年龄 [batch_size] []
            user_job: 用户职业 [batch_size] []
            movie_id: 候选电影ID [batch_size] []
            movie_categories: 候选电影类别 [batch_size] [18]
            movie_titles: 候选电影标题 [batch_size] [18]
            history_movie_ids: 历史交互电影ID [batch_size, hist_len] [2314]
        """

        # 嵌入用户特征
        uid_embed = self.uid_embedding(uid)
        gender_embed = self.gender_embedding(user_gender)
        age_embed = self.age_embedding(user_age)
        job_embed = self.job_embedding(user_job)

        # 嵌入候选电影特征
        movie_id_embed = self.movie_id_embedding(movie_id)
        movie_categories_embed = self.movie_categories_embedding(movie_categories)
        movie_titles_embed = self.movie_title_embedding(movie_titles)

        # 嵌入历史电影ID特征
        hist_movie_embed = self.history_movie_embedding(history_movie_ids)

        # 注意力机制处理历史交互
        attention_input = torch.cat([
            movie_id_embed.unsqueeze(1),
            hist_movie_embed
        ], dim=1)

        # 扩展最后一个维度
        attention_input_expanded = torch.cat([attention_input, attention_input], dim=-1)

        # 调整其维度
        batch_size, seq_len, _ = attention_input_expanded.shape
        embedding_dim = 16
        attention_input_reshaped = attention_input_expanded.view(-1, embedding_dim * 2)
        attention_weight = self.attention(attention_input_reshaped)

        # 恢复形状
        attention_output = attention_weight.view(batch_size, seq_len, 1)

        # 调整 attention_output 的维度，使其与 hist_movie_embed 匹配
        attention_output = attention_output[:, 1:, :]
        hist_attention = torch.sum(attention_output * hist_movie_embed, dim=1)

        # 这里的特征维度一定要对齐
        movie_categories_embed_mean = torch.mean(movie_categories_embed, dim=1)
        movie_titles_embed_mean = torch.mean(movie_titles_embed, dim=1)


        # print("uid_embed维度:", uid_embed.size())
        # print("gender_embed维度:", gender_embed.size())
        # print("age_embed维度:", age_embed.size())
        # print("job_embed维度:", job_embed.size())
        # print("movie_id_embed维度:", movie_id_embed.size())
        # print("movie_categories_embed_mean维度:", movie_categories_embed_mean.size())
        # print("movie_titles_embed_mean维度:", movie_titles_embed_mean.size())
        # print("hist_attention:", hist_attention.size())
        concat_features = torch.cat([
            uid_embed,  # [32, 16]
            gender_embed, # [32, 16]
            age_embed, # [32, 16]
            job_embed, # [32, 16]
            movie_id_embed,  # [32, 16]
            movie_categories_embed_mean,  # [32, 16]
            movie_titles_embed_mean, # [32, 16]
            hist_attention # [32, 16]
        ], dim=1)  #【32, 128]

        # 输出预测分数
        return self.prediction(concat_features) # [32, 1] 32为一个批次，其中每个值都代表一个预测分数