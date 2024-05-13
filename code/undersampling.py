import numpy as np
import pandas as pd

# 假设 X 是你的特征矩阵，y 是对应的标签
# X 和 y 应该是numpy数组或pandas数据帧
#y=pd.read_csv('4_train_label.csv',header=None)
#y=np.array(y)
#X=pd.read_csv('E:/Desktop/毕设/Pse-in-One-1.0.6/Pse-in-One/pse_3_4_train_seq.csv',header=None)
#X=np.array(X)

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn import under_sampling
from collections import Counter
from sklearn.utils import shuffle






def custom_random_undersampling(file_name,X1, X2,y, target_ratio=1.0, random_seed=None):
    # 随机洗牌
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(y))
    X1_shuffled = X1[shuffled_indices]
    X2_shuffled = X2[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    y_shuffled=y_shuffled.astype(int)
    # 计算多数类别（负样本）和少数类别（正样本）的样本数量
    neg_indices = np.where(y_shuffled == 0)[0]
    pos_indices = np.where(y_shuffled == 1)[0]

    neg_class_count = len(neg_indices)
    pos_class_count = len(pos_indices)

    # 计算目标下采样的负样本数量
    target_neg_count = int(pos_class_count * target_ratio)

    # 随机选择负样本的索引
    selected_neg_indices = np.random.choice(neg_indices, size=target_neg_count, replace=False)

    # 合并选定的负样本和所有正样本的索引
    selected_indices = np.concatenate((selected_neg_indices, pos_indices))
    np.savetxt(file_name+'_indx.txt',selected_indices,fmt="%d")
    # 使用选定的索引提取对应的特征和标签
    X1_resampled = X1_shuffled[selected_indices]
    X2_resampled = X2_shuffled[selected_indices]
    y_resampled = y_shuffled[selected_indices]

    return X1_resampled, X2_resampled,y_resampled


# 使用自定义下采样函数
#X_resampled, y_resampled = custom_random_undersampling(X, y, target_ratio=1.0, random_seed=42)

# 打印下采样后的类别分布
#print("Class distribution after resampling:", np.bincount(y_resampled))





def enn_sampling(X,y):
    # 创建EditedNearestNeighbours对象
    y = y.astype(int)
    # 将三维数组展平成二维数组
    X_flat = X.reshape((X.shape[0], -1))

    # 创建EditedNearestNeighbours对象
    enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3)

    # 进行ENN采样
    X_resampled_flat, y_resampled = enn.fit_resample(X_flat, y)
    # 获取被保留的样本在原始数据中的索引
    preserved_indices = enn.sample_indices_

    # 将采样后的数据还原成三维数组
    X_resampled = X_resampled_flat.reshape((X_resampled_flat.shape[0], X.shape[1], X.shape[2]))



    return X_resampled,y_resampled,preserved_indices


def cluster_sampling(X, y):
    X_flat = X.reshape((X.shape[0], -1))
    y = y.astype(int)
    cc = under_sampling.ClusterCentroids(sampling_strategy='auto')
    X_resampled_flat, y_resampled = cc.fit_resample(X_flat, y)

    # Get the indices of the retained samples
    retained_indices = cc.sample_indices_
    X_resampled = X_resampled_flat.reshape((X_resampled_flat.shape[0], X.shape[1], X.shape[2]))
    return X_resampled, y_resampled, retained_indices




def cnn_sampling(X,y):
    X_flat = X.reshape((X.shape[0], -1))
    cc = under_sampling.CondensedNearestNeighbour()
    X_resampled_flat, y_resampled = cc.fit_resample(X_flat, y)

    # Get the indices of the retained samples
    retained_indices = cc.sample_indices_
    X_resampled = X_resampled_flat.reshape((X_resampled_flat.shape[0], X.shape[1], X.shape[2]))
    return X_resampled, y_resampled, retained_indices


def tomolinks_sampling(X,y):
    y=y.astype(int)
    X_flat = X.reshape((X.shape[0], -1))
    tl=under_sampling.TomekLinks()
    X_resampled_flat, y_resampled = tl.fit_resample(X_flat, y)
    indices=tl.sample_indices_
    X_resampled = X_resampled_flat.reshape((X_resampled_flat.shape[0], X.shape[1], X.shape[2]))
    return  X_resampled,y_resampled,indices