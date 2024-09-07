import time

import torch
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

from application.decision_tree.decision_tree import DecisionTree, encode

# ----------------数据准备----------------------------
iris = load_iris()  # 加载数据
# ---------------模型训练----------------------------------
clf = tree.DecisionTreeClassifier(criterion="gini",
                                  splitter="best",
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.,
                                  max_features=None,
                                  random_state=None,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.,
                                  class_weight=None,
                                  ccp_alpha=0.0)  # sk-learn的决策树模型
clf = clf.fit(iris.data, iris.target)  # 用数据训练树模型构建()
r = tree.export_text(clf, feature_names=iris['feature_names'])  # 训练好的决策树
# ---------------模型预测结果------------------------
text_x = iris.data[[0, 1, 50, 51, 100, 101], :]
pred_target_prob = clf.predict_proba(text_x)  # 预测类别概率
pred_target = clf.predict(text_x)  # 预测类别

# ---------------打印结果---------------------------
print("\n===模型======")
print(r)
print("\n===测试数据：=====")
print(text_x)
print("\n===预测所属类别概率：=====")
print(pred_target_prob)
print("\n===预测所属类别：======")
print(pred_target)

print("================================")
print(clf.get_depth())
print(clf.tree_.children_left)
print(clf.tree_.children_right)
print(clf.tree_.feature)
print(clf.tree_.threshold)
print(clf.tree_.value)
print(clf.classes_)
print(clf.tree_.n_features)

dt = DecisionTree()

depth, tree = encode(clf)
print(tree)
# print(dt.tree)
print(type(clf))
print(type(clf.tree_))

# l = clf.tree_.children_left
# print(l)
# data = np.where(l == -1, np.arange(len(l)), l)
# print(data)
#
# value = clf.tree_.value
# max_indices = value.argmax(axis=2).flatten()

# feature = clf.tree_.feature
# print(feature)
# result = np.zeros((len(feature), clf.tree_.n_features), dtype=int)
#
# # 应用转换规则
# # 将2转换为[0, 0, 1, 0]，3转换为[0, 0, 0, 1]
# for i, val in enumerate(feature):
#     result[i][val] = 1
#
# print(result)
# print(max_indices)
#
# a = torch.tensor([1, 2, 3, 4, 5])
# b = torch.tensor([2, 3, 4, 5, 6])
#
# print(a.dot(b))

# 5
# [ 1 -1  3  4  5 -1 -1  8 -1 10 -1 -1 13 14 -1 -1 -1]
# [ 2 -1 12  7  6 -1 -1  9 -1 11 -1 -1 16 15 -1 -1 -1]
# [ 3 -2  3  2  3 -2 -2  3 -2  0 -2 -2  2  0 -2 -2 -2]
# [ 0.80000001 -2.          1.75        4.95000005  1.65000004 -2.
#  -2.          1.55000001 -2.          6.94999981 -2.         -2.
#   4.85000014  5.95000005 -2.         -2.         -2.        ]
