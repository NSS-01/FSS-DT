# key gen:
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

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

def download_dataset():
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
    from NssMPC.config import DEVICE
    from ucimlrepo import fetch_ucirepo
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, KBinsDiscretizer
    '''iris'''
    iris = load_iris()
    X, y = iris.data, iris.target
    '''breast_cancer'''
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    "MNIST"
    mnist = load_digits()
    X, y = mnist.data, mnist.target
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    X = X.to_numpy()
    y = y.to_numpy()
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
    X_preprocessed = preprocessor.fit_transform(X)
    "skin"
    # fetch dataset
    skin_segmentation = fetch_ucirepo(id=229)

    # data (as pandas dataframes)
    X = skin_segmentation.data.features
    y = skin_segmentation.data.targets
    X = X.to_numpy()
    y = y.to_numpy()
    '''
       covertype
       '''
    # # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets
    X = X.to_numpy()
    y = y.to_numpy()
    num_bins = 7
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform', subsample=10000)
    X = binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)


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
        server_data = torch.tensor(server_data)
        client_data = torch.tensor(client_data)

        client_file_path= f'./data/{data_name}_client_data.pth'
        server_file_path= f'./data/{data_name}_server_data.pth'
        torch.save((m0,threshold,client_data),client_file_path )
        torch.save((m0,threshold,server_data), server_file_path )

    '''iris'''
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
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
    client_data, server_data = torch.tensor(client_data, device=DEVICE), torch.tensor(server_data,
                                                                       device=DEVICE)
    torch.save((m0,thresholds,client_data), './data/iris_client_data.pth')
    torch.save((m0,thresholds,server_data), './data/iris_server_data.pth')
    m0, thresholds, server_data =torch.load('./data/iris_server_data.pth')
    m0,thresholds,client_data  =torch.load('./data/iris_client_data.pth')
    '''breast_cancer'''
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2023)
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
    client_data, server_data = torch.tensor(client_data, device=DEVICE), torch.tensor(server_data,
                                                                                      device=DEVICE)
    torch.save((m0,thresholds,client_data),'./data/breast_cancer_client_data.pth')
    torch.save((m0,thresholds,server_data),'./data/breast_cancer_server_data.pth')
    # data0 = torch.load('./data/breast_cancer_client_data.pth')
    # data1 = torch.load('./data/breast_cancer_server_data.pth')
    # print(data0,data1)
    "MNIST"
    mnist = load_digits()
    X, y = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2023)
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
    torch.save((m0,thresholds,client_data), './data/mnist_client_data.pth')
    torch.save((m0,thresholds,server_data), './data/mnist_server_data.pth')

    ''''heart_disease'''
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    X =  X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y == 0, test_size=0.2, random_state=2023)
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    save(thresholds, data, 'bank_marketing')
    # data0 = torch.load('./data/heart_disease_client_data.pth')
    # data1 =torch.load('./data/heart_disease_server_data.pth')
    # print(data1)

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
    y= label_encoder.fit_transform(y)
    # Step 1: 找到分类变量（字符串）和数值变量的列
    categorical_cols = X.select_dtypes(include=['object']).columns
    # print(X[categorical_cols])
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),  # 标准化数值型数据
            ('cat',  OneHotEncoder(), categorical_cols)  # One-Hot 编码字符串数据
        ])
    X_preprocessed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split( X_preprocessed, y, test_size=0.2, random_state=2023)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    save(thresholds, data, 'skin_segmentation')


    '''
    covertype
    '''
    # # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets
    X = X.to_numpy()
    y = y.to_numpy()
    num_bins = 2
    binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
    X= binner.fit_transform(X)
    # 使用 VarianceThreshold 移除常量特征
    selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
    X_reduced = selector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=2023)
    data = np.hstack((X_train, y_train.reshape(-1, 1)))
    thresholds = unique_values_by_column(torch.tensor(data))
    save(thresholds,data,'covertype')
    # data0 = torch.load('./data/covertype_client_data.pth')
    # data1 = torch.load('./data/covertype_server_data.pth')
if __name__ == '__main__':
    gen_key()
    config_dataset()
