# key gen:
import os.path
import warnings

import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, KBinsDiscretizer
from ucimlrepo import fetch_ucirepo
warnings.filterwarnings("ignore")

from NssMPC.crypto.aux_parameter import *
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams
from NssMPC.crypto.aux_parameter.truncation_keys.ass_trunc_aux_param import AssTruncAuxParams
def gen_key():
    gen_num = 1000

    AssMulTriples.gen_and_save(gen_num, saved_name='2PCBeaver', num_of_party=2)
    BooleanTriples.gen_and_save(gen_num, num_of_party=2)

    Wrap.gen_and_save(gen_num)
    GrottoDICFKey.gen_and_save(gen_num)
    RssMulTriples.gen_and_save(gen_num)
    DICFKey.gen_and_save(gen_num)
    SigmaDICFKey.gen_and_save(gen_num)
    ReciprocalSqrtKey.gen_and_save(gen_num)
    DivKey.gen_and_save(gen_num)
    GeLUKey.gen_and_save(gen_num)
    RssTruncAuxParams.gen_and_save(gen_num)
    B2AKey.gen_and_save(gen_num)
    AssTruncAuxParams.gen_and_save(gen_num)
#


# data processing
def config_dataset():
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
    from NssMPC.config import DEVICE
    from ucimlrepo import fetch_ucirepo
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, KBinsDiscretizer
    def unique_values_by_column(matrix):
        # print(matrix.shape)
        unique_vals = [torch.unique(matrix[:, i]).unsqueeze(-1) for i in range(matrix.size(1))]
        # torch.cat(unique_vals, dim=1)
        return unique_vals
    def save(threshold,data,data_name):
        m = len(data[0])
        m0 = m // 2
        # print(data[:, 0:m0])
        # print(data[:, m0:])
        client_data = np.zeros(data.shape)
        client_data[:, 0:m0] = data[:, 0:m0]
        server_data = np.zeros(data.shape)
        server_data[:, m0:] = data[:, m0:]
        server_data = torch.tensor(server_data,dtype=torch.int64)
        client_data = torch.tensor(client_data,dtype=torch.int64)

        client_file_path= f'./data/{data_name}_client_data.pth'
        server_file_path= f'./data/{data_name}_server_data.pth'
        torch.save((m0,threshold,client_data),client_file_path )
        torch.save((m0,threshold,server_data), server_file_path )

    '''iris'''
    iris = load_iris()
    X, y = iris.data, iris.target
    num_bins = 18
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=2023)
    torch.save((X_train, X_test, y_train, y_test), './data/iris.pth')
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    m = len(data[0])
    m0 = m // 2
    # print(data[:, 0:m0])
    # print(data[:, m0:])
    client_data = np.zeros(data.shape)
    client_data[:,0:m0]=data[:,0:m0]
    server_data = np.zeros(data.shape)
    server_data[:,m0:]=data[:,m0:]

    client_data, server_data = torch.tensor(client_data, dtype=torch.int64), torch.tensor(server_data,
                                                                       dtype=torch.int64)
    torch.save((m0,thresholds,client_data), './data/iris_client_data.pth')
    torch.save((m0,thresholds,server_data), './data/iris_server_data.pth')
    '''breast_cancer'''
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    num_bins = 4
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=2028)
    torch.save((X_train, X_test, y_train, y_test), './data/breast_cancer.pth')
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))

    m = len(data[0])

    m0 = m // 2
    # print(data[:, 0:m0])
    # print(data[:, m0:])
    client_data = np.zeros(data.shape)
    client_data[:, 0:m0] = data[:, 0:m0]
    server_data = np.zeros(data.shape)
    server_data[:, m0:] = data[:, m0:]
    client_data, server_data = torch.tensor(client_data, dtype=torch.int64), torch.tensor(server_data,
                                                                                          dtype=torch.int64)
    torch.save((m0,thresholds,client_data),'./data/breast_cancer_client_data.pth')
    torch.save((m0,thresholds,server_data),'./data/breast_cancer_server_data.pth')


    "MNIST"
    mnist = load_digits()
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2020)
    torch.save((X_train, X_test, y_train, y_test), './data/mnist.pth')
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))

    m = len(data[0])
    m0 = m // 2
    # print(data[:, 0:m0])
    # print(data[:, m0:])
    client_data = np.zeros(data.shape)
    client_data[:, 0:m0] = data[:,0:m0]
    server_data = np.zeros(data.shape)
    server_data[:, m0:] = data[:, m0:]
    client_data, server_data = torch.tensor(client_data, dtype=torch.int64), torch.tensor(server_data,
                                                    dtype=torch.int64)

    torch.save((m0,thresholds,client_data), './data/mnist_client_data.pth')
    torch.save((m0,thresholds,server_data), './data/mnist_server_data.pth')

    ''''heart_disease'''
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    X = X.dropna()  # Removes rows with any missing values in X
    y = y.loc[X.index]  # Ensure y matches the cleaned X rows
    X = X.to_numpy()
    y = y.to_numpy()
    num_bins = 12
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y == 0, test_size=0.2, random_state=2023)
    torch.save((X_train, X_test, y_train, y_test), './data/heart_disease.pth')
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    save(thresholds, data, 'heart_disease')

    '''bank_marketing '''
    # fetch dataset
    bank_marketing = fetch_ucirepo(id=222)
    # data (as pandas dataframes)
    bank_marketing.data
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    X = X.dropna()  # Removes rows with any missing values in X
    y = y.loc[X.index]  # Ensure y matches the cleaned X rows
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Step 1: 找到分类变量（字符串）和数值变量的列
    categorical_cols = X.select_dtypes(include=['object']).columns
    # print(X[categorical_cols])
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),  # 标准化数值型数据
            ('cat', OneHotEncoder(), categorical_cols)  # One-Hot 编码字符串数据
        ])
    X = preprocessor.fit_transform(X)
    num_bins = 4
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=2023)
    torch.save((X_train, X_test, y_train, y_test), './data/bank_marketing.pth')
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    save(thresholds, data, 'bank_marketing')
    # data0 = torch.load('./data/bank_marketing_client_data.pth')
    # data1 = torch.load('./data/bank_marketing_server_data.pth')
    # print(data0.shape,data1.shape)

    "skin"
    # fetch dataset
    skin_segmentation = fetch_ucirepo(id=229)

    # data (as pandas dataframes)
    X = skin_segmentation.data.features
    y = skin_segmentation.data.targets
    X = X.to_numpy()
    y = y.to_numpy()
    num_bins = 8
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y == 0, test_size=0.2, random_state=2031)
    torch.save((X_train, X_test, y_train, y_test), './data/skin.pth')
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    save(thresholds, data, 'skin')



    '''
    covertype
    '''

    covertype = fetch_ucirepo(id=31)
    X = covertype.data.features
    y = covertype.data.targets
    X = X.to_numpy()
    y = y.to_numpy()

    num_bins = 53
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="uniform")
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征

    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=1021)
    torch.save((X_train, X_test, y_train, y_test), './data/covertype.pth')
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    save(thresholds,data,'covertype')
if __name__ == '__main__':
    from pathlib import Path
    data_path = "./data"
    model_path = "./model"
    if not os.path.exists(data_path):
        Path(data_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(model_path):
        Path(model_path).mkdir(parents=True,exist_ok=True)

    config_dataset()

    # download_dataset()
    gen_key()
