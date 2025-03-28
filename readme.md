目前的进度：
1. 实现了用评分作为监督信号的模型，并且模型训练效果看起来还不错
（1）模型效果：/Users/linjiaxi/Desktop/MovieLens Recommendation System/runs/movie_recommendation_logs/events.out.tfevents.1741613656.linjiaxideMacBook-Air.local.63303.0
（2）模型路径：/Users/linjiaxi/Desktop/MovieLens Recommendation System/model_save/model.pth

待实现：
1. 基于对比学习（让负样本的值接近于0，正样本的值接近于1）的模型
2. 负样本的采样策略可以优化一下：多策略组合

思路：
1. 双塔召回模型（可以给定用户ID，召回相应的电影）