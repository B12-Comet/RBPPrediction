import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter
from sklearn.utils import shuffle

# 读取数据集
data = pd.read_csv('AAC_smote_data.csv')

# 提取特征和标签
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签

# 使用 ENN 下采样
enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3)
X_resampled, y_resampled = enn.fit_resample(X, y)

# 将采样后的数据转为 DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['label'] = y_resampled

# 查看采样后样本的类别分布
print(Counter(y_resampled))

# 打乱正负样本的分布
resampled_data = shuffle(resampled_data)

# 保存采样后的文件
resampled_data.to_csv('AAC_smote_ENN1.csv', index=False)
