import urllib.request
import pandas as pd
import requests
import zipfile
import os
import pickle
import re
from tqdm import tqdm

def download_data():
    # 定义下载的 URL 和保存的文件名
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    filename = './data/ml-1m.zip'
    extract_dir = './data/ml-1m'

    # 下载文件
    def download_file(url, filename):
        if os.path.exists(filename):
            print(f"{filename} 已存在，跳过下载。")
            return
        print(f"开始下载 {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(filename, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))
        print(f"{filename} 下载完成。")

    # 解压文件
    def extract_zip_file(filename, extract_dir):
        if os.path.exists(extract_dir):
            print(f"{extract_dir} 目录已存在，跳过解压。")
            return
        print(f"开始解压 {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"{filename} 解压完成。")

    download_file(url, filename)
    extract_zip_file(filename, extract_dir)
    print("数据下载并解压完成！")

def get_data():
    # 定义列名
    movies_cols = ['movie_id', 'title', 'genres']
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    users_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']

    # 读取.dat文件
    movies_df = pd.read_csv('./data/ml-1m/movies.dat', sep='::', names=movies_cols, engine='python', encoding='latin-1')
    ratings_df = pd.read_csv('./data/ml-1m/ratings.dat', sep='::', names=ratings_cols, engine='python', encoding='latin-1')
    users_df = pd.read_csv('./data/ml-1m/users.dat', sep='::', names=users_cols, engine='python', encoding='latin-1')

    # 转换为.csv文件
    movies_df.to_csv('./data/ml-1m/movies.csv', index=False)
    ratings_df.to_csv('./data/ml-1m/ratings.csv', index=False)
    users_df.to_csv('./data/ml-1m/users.csv', index=False)
    print("数据保存为csv文件成功！")
    return movies_df, ratings_df, users_df


def preprocess_users_data(users_df):
    # 处理gender部分，F为1，M为0
    gender_mapping={'F':1, 'M':0}
    users_df['gender']=users_df['gender'].map(gender_mapping)

    # 处理age部分
    age_map={val: i for i, val in enumerate(set(users_df['age']))}
    users_df['age']=users_df['age'].map(age_map)
    print(f"用户数量为{len(users_df)}")
    print("users_df预处理成功！")
    return users_df


def preprocess_movies_data(movies_df):
    # （1）处理title中的年份
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(movies_df['title']))}
    movies_df['title'] = movies_df['title'].map(title_map)

    # (2) 电影title转换为数字字典
    title_set=set()
    for val in movies_df['title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int={val: i for i, val in enumerate(title_set)}

    #（3）将电影title转成等长列表，长度为15
    title_count=15
    # 遍历movies_df['title']列中所有唯一的电影标题，对于每个电影标题，将其按照空格分割成单词，然后通过title2int字典将每个单词转换为对应的整数
    title_map = {val: [title2int[row] for row in val.split()] for val in movies_df['title']}
    for key in title_map:
        for cnt in range(title_count-len(title_map[key])):
            title_map[key].insert(len(title_map[key])+cnt, title2int['<PAD>'])
    movies_df['title']=movies_df['title'].map(title_map)

    # (4) 电影类型转换为数字字典
    all_genres=set()
    for genre_str in movies_df['genres']:
        all_genres.update(genre_str.split('|'))
    all_genres.add('<PAD>')
    genres2int={val:i for i, val in enumerate(all_genres)}

    # (5) 电影类型转换为等长数字列表，长度是15
    genres_map = {val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies_df['genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values())-len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key])+cnt, genres2int['<PAD>'])
        
    movies_df['genres']=movies_df['genres'].map(genres_map)
    print(f"电影的类别数为{len(all_genres)}")
    print(f"电影的类别为{all_genres}")

    print("movies_df预处理成功！")
    return title2int, genres2int, title_map, title_set, title_count, movies_df, genres_map

def get_x_y(data):
    target_field=['rating']
    features_pd, targets_pd=data.drop(target_field, axis=1), data[target_field]
    features=features_pd.values
    targets_values=targets_pd.values
    print("特征列和目标列处理成功！")
    return features_pd, targets_pd, features, targets_values


if __name__=='__main__':
    # # 1. 下载数据
    # print("开始下载数据!")
    # download_data()

    # 2. 保存数据
    movies_df, ratings_df, users_df=get_data()

    # 3. 处理用户数据
    users_df=preprocess_users_data(users_df)

    # 4. 处理电影数据
    title2int, genres2int, title_map, title_set, title_count, movies_df, genres_map=preprocess_movies_data(movies_df)

    # 5. 处理评分数据
    ratings = ratings_df.filter(regex='user_id|movie_id|rating')

    # 6. 合并三个表
    data=pd.merge(pd.merge(ratings, users_df), movies_df)

    # 7. 获取X和y两张表
    features_pd, targets_pd, features, targets_values=get_x_y(data)

    # 8. 保存数据到本地
    pickle.dump((title2int, title_count, title_set, genres2int, genres_map, features_pd, targets_pd, features, targets_values, ratings_df, users_df, movies_df, data),open('./data/preprocess.p', 'wb'))
    print("相关数据保存到本地成功！")

    # 9. 保存预处理后的数据到本地
    users_df.to_csv('./data/preprocess_data/process_users_df.csv')
    movies_df.to_csv('./data/preprocess_data/process_movies_df.csv')
    data.to_csv('./data/preprocess_data/data.csv')
    print("预处理后的数据保存成功！")