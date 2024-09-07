import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder,KBinsDiscretizer
from ucimlrepo import fetch_ucirepo



from  config.base_configs import  DEVICE
Maximum_depth = 4 # h = Max

'''
define  secure  tree list struct 
'''
class TreeNode(object):
    def __init__(self, t=-1, m=-1):
        self.m = m  # 存储某种属性m
        self.t = t  # 存储另一种属性t

    def __str__(self):
        # 返回想要显示的信息，这里展示了如何将属性m和t整合到一条字符串中
        return f"TreeList Object: m={self.m}, t={self.t}"

def hash_function(x, N):
    hashed_x = x * N
    return hashed_x.astype(int)  # Convert to integer
def unique_values_by_column(matrix):
    unique_vals = [torch.unique(matrix[:, i]).unsqueeze(1) for i in range(matrix.size(1))]
    return unique_vals

# Apply the hash function to each column of X

'''
init a training samples of decision tree
'''
# datasets = []

# iris = load_iris()
# X, y = iris.data, iris.target
#
# torch.save((X,y),"iris.pth")
X, y = torch.load("application/Lightweiht_disicion_tree/iris.pth")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.reshape(-1, 1)))
# datasets.append(data)
#
# breast_cancer = load_breast_cancer()
# X, y = breast_cancer.data, breast_cancer.target
# N = 1  # Assume N is 1000, or any other large number you'd like to use
# hashed_X = np.apply_along_axis(hash_function, 0, X, N)
# torch.save((X,y),"breast_cancer.pth")
X, y = torch.load("application/Lightweiht_disicion_tree/breast_cancer.pth")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.reshape(-1, 1)))

''''heart_disease'''
from ucimlrepo import fetch_ucirepo

# # fetch dataset
# heart_disease = fetch_ucirepo(id=45)
#
#
# # data (as pandas dataframes)
# X = heart_disease.data.features
# y = heart_disease.data.targets
#
# X =  X.to_numpy()
# y = y.to_numpy()

# torch.save(
#     (X,y>0),"heart_disease.pth"
# )
X, y = torch.load("application/Lightweiht_disicion_tree/heart_disease.pth")
X_train, X_test, y_train, y_test = train_test_split(X, y==0, test_size=0.2, random_state=2023)


'''rice '''
# from ucimlrepo import fetch_ucirepo
#
# # fetch dataset
# rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
#
# # data (as pandas dataframes)
# X = rice_cammeo_and_osmancik.data.features
# y = rice_cammeo_and_osmancik.data.targets
# X = X.to_numpy()
# y = (y!=0).to_numpy()
# torch.save((X,y),"rice.pth")
X, y = torch.load("application/Lightweiht_disicion_tree/rice.pth")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
data = np.hstack((X_train, y_train.reshape(-1, 1)))



# from ucimlrepo import fetch_ucirepo
# '''bank_marketing '''
# # fetch dataset
# bank_marketing = fetch_ucirepo(id=222)
#
# # data (as pandas dataframes)
# bank_marketing.data
# X = bank_marketing.data.features
# y = bank_marketing.data.targets
# X = X.dropna()  # Removes rows with any missing values in X
# y = y.loc[X.index]  # Ensure y matches the cleaned X rows
# label_encoder = LabelEncoder()
# y= label_encoder.fit_transform(y)
# # Step 1: 找到分类变量（字符串）和数值变量的列
# categorical_cols = X.select_dtypes(include=['object']).columns
# # print(X[categorical_cols])
# numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_cols),  # 标准化数值型数据
#         ('cat',  OneHotEncoder(), categorical_cols)  # One-Hot 编码字符串数据
#     ])
# X_preprocessed = preprocessor.fit_transform(X)
# torch.save((X_preprocessed,y),"bank_market.pth")
X, y = torch.load("application/Lightweiht_disicion_tree/bank_market.pth")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
data = np.hstack((X_train, y_train.reshape(-1, 1)))

'''load_digits'''
# digits = load_digits()
# y = digits.target
# X = digits.data
# # X, y = iris.data, iris.target
#
# torch.save((X,y),"digits.pth")
#
X, y = torch.load("application/Lightweiht_disicion_tree/digits.pth")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.reshape(-1, 1)))

# "fetch_ucirepo"

from ucimlrepo import fetch_ucirepo

# fetch dataset
# skin_segmentation = fetch_ucirepo(id=229)
#
# # data (as pandas dataframes)
# X = skin_segmentation.data.features
# y = skin_segmentation.data.targets
# X = X.to_numpy()
# y = y.to_numpy()
# torch.save((X,y),"skin_segmentation.pth")
X, y = torch.load("application/Lightweiht_disicion_tree/skin_segmentation.pth")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.reshape(-1, 1)))

'''covertype
'''
#
# from ucimlrepo import fetch_ucirepo
#
# # fetch dataset
# covertype = fetch_ucirepo(id=31)
#
# # data (as pandas dataframes)
# X = covertype.data.features
# y = covertype.data.targets
# X = X.to_numpy()
# y = y.to_numpy()
# num_bins = 7
# binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform',subsample=10000)
# X= binner.fit_transform(X)
# # 使用 VarianceThreshold 移除常量特征
# selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
# X_reduced = selector.fit_transform(X)
# torch.save((X_reduced[0:10000],y[0:10000]),"covertype_data_first.pth")
# torch.save((X_reduced[10000:],y[10000:]),"covertype_data_remain.pth")

X1, y1 = torch.load("application/Lightweiht_disicion_tree/covertype_data_first.pth")
X2, y2 = torch.load("application/Lightweiht_disicion_tree/covertype_data_remain.pth")

X = np.concatenate((X1, X2))
y = np.concatenate((y1,y2))

# print(len(X_reduced[0]))
# num_features_to_select = 40
# selected_feature_indices = np.random.choice(X.shape[1], size=num_features_to_select, replace=False)
# X = X[:, selected_feature_indices]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
data = np.hstack((X_train, y_train.reshape(-1, 1)))
'''
init  the data of $P_0$ and $P_1$ 
'''
m = len(data[0])
m0 = m//2
# data = data[1:10000] if len(data)>10*4 else data
client_data, server_data = torch.tensor(data,dtype=torch.int,device= DEVICE), torch.tensor(data,dtype=torch.int,device= DEVICE)

'''
pick thresholds function
'''

# def sum_of_unique_values_count_by_column(matrix):
#     # 计算每列的唯一值数量
#     unique_counts = [torch.unique(matrix[:, i]).size(0) for i in range(matrix.size(1))]
#     # 返回唯一值数量的总和
#     return sum(unique_counts)
# for f in datasets:
#     print(f"ts numbers:{sum_of_unique_values_count_by_column(torch.tensor(f))}")



