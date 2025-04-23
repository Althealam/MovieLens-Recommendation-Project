## 代码结构
1. recall：召回模块
* movie_tower.py：电影塔
* user_tower.py：用户塔
* two_tower_model.py：双塔模型
* dataset.py：双塔模型数据处理
* train.py：双塔模型训练脚本
* all_code.py：所有代码集成的脚本
* debug_two_tower_model.ipynb：双塔模型调试代码
2. rank：排序模块
* din_model.py：DIN模型
* attention_layer.py：注意力层
* activation_function：自定义激活函数
* dice.py：Dice激活函数
* rank_model：排序模型
* dataset.py：排序模型数据处理
* train.py：排序模型训练脚本
* debug_rank_model_v1.ipynb：Dice模型调试代码 version1
* debug_rank_model_v2.ipynb：Dice模型调试代码 version2
3. data.py：获取MovieLens的数据


## 更新日志
【Update 2025/3/31】
1. 实现了用评分作为监督信号的模型，并且模型训练效果看起来还不错
（1）模型效果：/Users/linjiaxi/Desktop/MovieLens Recommendation System/runs/movie_recommendation_logs/events.out.tfevents.1741613656.linjiaxideMacBook-Air.local.63303.0
（2）模型路径：/Users/linjiaxi/Desktop/MovieLens Recommendation System/model_save/model.pth
2. 实现了用对比学习作为训练策略的模型，但是模型效果一般
其中将评分大于3分的样本作为正样本，小于等于3分的作为负样本
得到的效果是train和test的时候loss都大概为5.5左右
模型代码路径：two_tower_model_v1.py
问题：负样本的选取不太好
3. 优化负样本的获取策略，看起来模型效果不错，一直处于下降趋势
负样本选取策略：
（1）同一batch内其他样本作为负样本
（2）使用难负样本挖掘策略，选择相似度较高的负样本
（3）使用热门电影作为负样本
模型代码路径：two_tower_model_v2.py
问题：
（1）模型做测试的时候效果不算太好，测试效果大概是4.1左右（打算先就这样了，后续再优化）
（2）热门电影的选取有问题（涉及到负样本）

【Update 2025/4/1】
1. 完成了召回模型代码的注释部分
2. 修改了召回模型热门电影的获取方法（之前是直接从movies_df中选择前面的1000个，现在改成了统计每部电影的出现频次，然后选择出现频次最高的前1000个）
* 问题：发现训练一个epoch后，loss就一直卡在4.1148下不来（最终的loss是4.1155）
* 可能的优化点：目前是pointwise的训练方式，也就是独立的考虑一个正样本和一个负样本。但是获取我们可以考虑listwise, pairwise的训练方式。同时，目前的模型存在问题还有用户和物品的特征向量无法交互，可以改进（参考美团的对偶增强向量模型）
3. 初步写了排序模型的GBDT代码，但是还没跑通
* 代码路径：debug_rank_model_v1.ipynb（这个是debug代码）

【Update 2025/4/2】
1. 初步写了排序模型的DIN代码（GBDT代码跑不通，因为需要全都是数值型，先暂时放弃了）
* 问题：目前DIN代码还存在报错，需要debug下

【Update 2025/4/7】
1. DIN排序模型已经跑通了，但是Loss超级高，需要debug一下看看问题在哪
初步怀疑是那些乱七八糟的维度问题导致的

【Update 2025/4/10】
1. 检查了一下DIN模型，对特征维度做了一些些修改，但是Loss还是超级高，需要debug看看问题在哪


【Update 2025/4/12】
1. 在train函数中打印了梯度值，发现有梯度爆炸的问题存在
（1）在attention层将ReLU激活函数替换为Dice，但是仍然有梯度爆炸的问题
2. 之前存在的问题
（1）train_loader在用enumerate输出的时候要按照顺序，之前的history_movie_ids和ratings的顺序反了，已经改过来了
（2）prediction层的维度必须要和concat_features的维度对齐，要让linear层的矩阵可以和concat_features相乘
3. 还存在的问题：
（1）prediction的预测值超过了5，并且还会有负数出现

【Update 2025/4/14】
1. 定义了一个自定义的激活函数，限制输出在1到5之间，但是Loss还是会出现震荡的情况，没办法收敛
* 代码路径：debug_rank_model_v1.ipynb
一些可能优化的点：
1. 处理侯选电影和历史电影的embedding，比如user_id, candidate_movie_id, rating, gender, age, occupation, title, genres, history_movie_id, history_movie_title, history_movie_genres
然后将history_movie_id, history_movie_title, history_movie_genres处理成embedding合并在一起
* 代码地址：DIN_dataset.ipynb

【Update 2025/4/15】
1. 弄了双塔模型的推理部分，希望可以实现对于每个用户ID，推荐最有可能感兴趣的top10个电影（优先级比较高的问题）
* 代码路径：two_tower_model_infer.ipynb
* 仍存在的问题：返回的相似度矩阵的元素全都是相同的，可能是在模型调用的地方出现了问题
2. 弄了DIN模型的第二版，感觉有以下问题需要注意：
（1）使用原本的movies.csv, ratings.csv, users.csv比较好，因为ratings里面有timestamp，这样可以获得序列关系
（2）DIN在加载数据的时候特别慢，跑了50多分钟还没跑出来，可能需要先用双塔召回后再来跑比较好，或者是先用验证集来弄，避免数据量太大了跑不出来。

【Update 2025/4/16】
1. 弄了双塔模型的推理部分，需要将所有电影的embedding缓存起来，目前正在弄这个，现在还有报错，需要解决
2. 后续需要基于给定的embedding和用户ID，返回该用户可能感兴趣的topk个电影，弄完召回后，再去弄DIN

【Update 2025/4/17】
1. 将rank和recall模块进行了划分
还需要做的事情：整理一下recall和rank部分的内容

【Update 2025/4/21】
1. 整理双塔模型的电影塔部分的内容到飞书上
待完成事项：在目前的双塔模型的基础上优化一下（用分支two_tower_v2），增加一些其他的特征（用户特征矩阵、电影特征矩阵）


【Update 2025/4/22】
1. 获取用户特征和电影特征，并且保存到data的features文件夹下
2. 构建双塔模型
待完成事项：跑通双塔模型的训练代码

【Update 2025/4/23】
1. 弄了一下user_features, movie_features：对title, genres做了处理；并且弄了user_dataset和movie_dataset，但是目前还存在报错，需要排查
