import joblib
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, load_diabetes
from sklearn import tree
import pandas as pd


def load_datasets(datasets):
    if datasets in ['iris', 'wine', 'breast_cancer', 'digits', 'diabetes']:
        data = eval(f'load_{datasets}()')
        return data.data, data.target
    elif datasets == 'boston':
        data = pd.read_csv('data/boston.csv')
        data = data.values
        x = data[:, :-1]
        y = (data[:, -1] * 10).astype(int)
        return x, y
    elif datasets == 'spambase':
        data = pd.read_csv('data/spambase.csv')
        data = data.values
        x = data[:, :-1]
        y = data[:, -1].astype(int)
        return x, y
    else:
        print('Unsupported datasets')


def train_model(data, target, max_depth=None, max_features=None):
    clf = tree.DecisionTreeClassifier(criterion="gini",
                                      splitter="best",
                                      max_depth=max_depth,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.,
                                      max_features=max_features,
                                      random_state=None,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0.,
                                      class_weight=None,
                                      ccp_alpha=0.0)  # sk-learn的决策树模型
    clf = clf.fit(data, target)
    return clf


def print_info(clf):
    print("depth", clf.get_depth())
    unique_feature = np.unique(clf.tree_.feature)
    num_of_used_feature = len(unique_feature)
    if -2 in unique_feature:
        num_of_used_feature -= 1
    print("used features", num_of_used_feature)
    print("total features", clf.tree_.n_features)
    print("nodes", clf.tree_.children_left.shape[0])


def train_datasets(datasets):
    max_depth = {'iris': None, 'wine': 5, 'breast_cancer': 7, 'digits': 15, 'diabetes': 28, 'boston': 30,
                 'spambase': 17}
    max_features = {'iris': None, 'wine': 7, 'breast_cancer': 12, 'digits': 47, 'diabetes': 10, 'boston': 13,
                    'spambase': 57}
    if datasets == "mnist":
        clf = joblib.load("debug/application/DT/mnist_dt_model.pkl")
    elif datasets in max_depth.keys():
        data, target = load_datasets(datasets)
        clf = train_model(data, target, max_depth=max_depth[datasets], max_features=max_features[datasets])
    else:
        print('Unsupported datasets')
        return
    print_info(clf)
    return clf
