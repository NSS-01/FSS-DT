import numpy as np
import torch
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from  config.base_configs import  DEVICE
Maximum_depth = 2 # h = Max

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
datasets = []

iris = load_iris()
X, y = iris.data, iris.target
N = 10**3  # Assume N is 1000, or any other large number you'd like to use
hashed_X = np.apply_along_axis(hash_function, 0, X, N)

X_train, X_test, y_train, y_test = train_test_split(hashed_X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.reshape(-1, 1)))
datasets.append(data)

iris = load_breast_cancer()
X, y = iris.data, iris.target
N = 10**3  # Assume N is 1000, or any other large number you'd like to use
hashed_X = np.apply_along_axis(hash_function, 0, X, N)

X_train, X_test, y_train, y_test = train_test_split(hashed_X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.reshape(-1, 1)))
datasets.append(data)


# samples = load_iris()
# samples =  load_digits()
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)


# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

N = 10**3  # Assume N is 1000, or any other large number you'd like to use
hashed_X = np.apply_along_axis(hash_function, 0, X, N)

X_train, X_test, y_train, y_test = train_test_split(hashed_X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.to_numpy().reshape(-1, 1)))
datasets.append(data)



# # data (as pandas dataframes)
# X = rice_cammeo_and_osmancik.data.features
# y = rice_cammeo_and_osmancik.data.target

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Numerical data imputation
X = X.dropna()  # Removes rows with any missing values in X
y = y.loc[X.index]  # Ensure y matches the cleaned X rows
# # iris  = load_digits()
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
le = LabelEncoder()

# Apply label encoding to the column that contains 'Besni' (let's say it's the first column for this example)
X= X.apply(le.fit_transform)

y = le.fit_transform(y)

# X, y = samples.data, samples.target
N = 10**3  # Assume N is 1000, or any other large number you'd like to use
hashed_X = np.apply_along_axis(hash_function, 0, X, N)

X_train, X_test, y_train, y_test = train_test_split(hashed_X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.reshape(-1, 1)))
datasets.append(data)

iris = load_digits()
X, y = iris.data, iris.target
N = 10**3  # Assume N is 1000, or any other large number you'd like to use
hashed_X = np.apply_along_axis(hash_function, 0, X, N)

X_train, X_test, y_train, y_test = train_test_split(hashed_X, y, test_size=0.2, random_state=2023)

data = np.hstack((X_train, y_train.reshape(-1, 1)))
datasets.append(data)

'''
init  the data of $P_0$ and $P_1$ 
'''
m = len(data[0])
m0 = 2
client_data, server_data = torch.tensor(data,dtype=torch.int64,device= DEVICE), torch.tensor(data,dtype=torch.int64,device= DEVICE)

'''
pick thresholds function
'''

def sum_of_unique_values_count_by_column(matrix):
    # 计算每列的唯一值数量
    unique_counts = [torch.unique(matrix[:, i]).size(0) for i in range(matrix.size(1))]
    # 返回唯一值数量的总和
    return sum(unique_counts)
for f in datasets:
    print(f"ts numbers:{sum_of_unique_values_count_by_column(torch.tensor(f))}")

#
# import torch

# Create example matrices
# A = torch.randn(3, 120)  # Matrix A with size [3, 120]
# B = torch.randn(120, 31) # Matrix B with size [120, 31]
#
# # Perform matrix multiplication
# C = torch.matmul(A, B)   # or C = A @ B
#
# print("The size of the resulting matrix C:", C.size())

#
# import torch
#
# # Example data (last column are the labels)
# data = torch.tensor([[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 3],
#                      [9, 10, 6]], dtype=torch.int64)
#
# # Unique labels (for illustration purposes, extracted manually here)
# unique_labels = torch.tensor([3, 6], dtype=torch.int64)
#
# # Reshape data labels and unique labels for broadcasting
# data_labels = data[:, -1].unsqueeze(1)  # Shape: (num_samples, 1)
# unique_labels = unique_labels.unsqueeze(0)  # Shape: (1, num_labels)
# print(data_labels)
#
# # Broadcasted comparison to generate binary matrix
# binary_matrix = data_labels == unique_labels  # Shape: (num_samples, num_labels)
#
# print(binary_matrix)


