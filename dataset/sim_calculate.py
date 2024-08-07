import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

matrix_path = 'data/KuaiRec 2.0/data/big_matrix.csv'
matrix = pd.read_csv(matrix_path)

# 修改兴趣分数
matrix[matrix["watch_ratio"]>2]=2
matrix["watch_ratio"] = matrix["watch_ratio"]/2


matrix = matrix.drop_duplicates(subset=['user_id','video_id'],keep='first')

base_name = os.path.basename(matrix_path)
train_path = os.path.join('data',base_name.replace(".csv","_train.npy"))
test_path = os.path.join('data',base_name.replace(".csv","_test.npy"))

# 按照时间戳划分数据集，25%采样为测试集
train_matrix = matrix[matrix['date']<20200830]
test_matrix = matrix[matrix['date']>=20200830]

# 计算user-items矩阵（排序后）
user_item_matrix_train = train_matrix.pivot(index='user_id', columns='video_id', values='watch_ratio').fillna(0)
user_item_matrix_test = test_matrix.pivot(index='user_id', columns='video_id', values='watch_ratio').fillna(0)

# 获取item_item的相似度矩阵
values_train = user_item_matrix_train.values.T
items_items_matrix_train = cosine_similarity(values_train,values_train)
values_test = user_item_matrix_test.values.T
items_items_matrix_test = cosine_similarity(values_test,values_test)

train_items = list(set(train_matrix["video_id"]))
test_items = list(set(test_matrix["video_id"]))
train_items.sort()
test_items.sort()
train_items = np.array(train_items).T
test_items = np.array(test_items).T
# 找到相似度最相似的物体排序
train_np = items_items_matrix_train.argsort(axis=-1)[:,::-1]
test_np = items_items_matrix_test.argsort(axis=-1)[:,::-1]
train_np[:,0] = train_items
test_np[:,0] = test_items
np.save(train_path,train_np)
np.save(test_path,test_np)