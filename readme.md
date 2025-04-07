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
