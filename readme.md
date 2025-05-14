## 仓库结构
1. config：包含各种配置文件，比如movie_embedding_config, user_embedding_config
2. data：经过负样本采样后的数据 比如pos_neg_data
3. embedding：存放经过电影塔的embedding向量
4. faiss：将经过电影塔的embedding向量以faiss的形式存储
5. features：包含用户和物品的特征向量（经过处理的）
6. 代码顺序：feature_engineering=>get_neg_pos_sample=>embedding_config=model

## 更新日志
【Update】20250514
1. 增加了SENet模块，发现训练一个epoch的时候的AUC和accuracy都有提升
2. pairwise训练代码，现在正在弄模型输入生成部分
可以优化的点：加一个补水塔