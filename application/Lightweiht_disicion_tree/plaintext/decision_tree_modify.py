import sys

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits,load_diabetes
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import time
import matplotlib.pyplot as plt
class DecisionTreeNode:
    """A decision tree node class for binary tree"""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        """
        - feature_index: Index of the feature used for splitting.
        - threshold: Threshold value for splitting.
        - left: Left subtree.
        - right: Right subtree.
        - value: Value of the node if it's a leaf node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if the node is a leaf node"""
        return self.value is not None


def build_tree(data, max_depth, depth=0):
    """Builds the decision tree recursively"""
    X, y = data[:, :-1], data[:, -1]
    num_samples, num_features = X.shape

    # If all data has the same label or max depth reached, return a leaf node
    if num_samples <= 1 or len(set(y)) == 1 or depth > max_depth:
        leaf_value = most_common_label(y)
        return DecisionTreeNode(value=leaf_value)

    # Else, find the best split
    feature_idx, threshold = optimized_find_best_split(data, num_features)
    # print(f"feature_idx:{feature_idx},threshold: {threshold}")
    # Split the dataset
    left_idx, right_idx = split_dataset(X[:, feature_idx], threshold)

    left_subtree = build_tree(data[left_idx, :], max_depth, depth + 1)
    right_subtree = build_tree(data[right_idx, :], max_depth, depth + 1)

    # Return decision node
    return DecisionTreeNode(feature_index=feature_idx, threshold=threshold, left=left_subtree, right=right_subtree)


def most_common_label(y):
    """Returns the most common label in y"""


    return max(set(y), key=list(y).count,default=0)


def number_squared_for_lbl(y):
    _, counts = np.unique(y, return_counts=True)
    return np.sum(counts ** 2)


def optimized_number_squared_for_lbl(y):
    """Optimized function to calculate the sum of squares of label counts."""
    unique, counts = np.unique(y, return_counts=True)
    return np.sum(counts ** 2)


def optimized_find_best_split(data, num_features):
    X, y = data[:, :-1], data[:, -1]
    best_feature_idx, best_threshold = 0, 0
    best_metric = float('inf')

    for feature_idx in range(num_features):
        thresholds = np.unique(X[:, feature_idx])
        # Pre-calculate metrics for all thresholds
        metrics = np.array([
            calculate_metric(X[:, feature_idx], y, threshold)
            for threshold in thresholds
        ])

        # Find the threshold with the best metric
        min_metric_idx = np.argmin(metrics)
        if metrics[min_metric_idx] < best_metric:
            best_metric = metrics[min_metric_idx]
            best_feature_idx, best_threshold = feature_idx, thresholds[min_metric_idx]

    return best_feature_idx, best_threshold


def calculate_metric(feature_column, y, threshold):
    """Calculates a metric (like Gini index) for a given feature and threshold."""
    left_mask = feature_column <= threshold
    right_mask = ~left_mask
    y_left, y_right = y[left_mask], y[right_mask]

    if len(y_left) == 0 or len(y_right) == 0:
        return  sys.maxsize

    DL, DR = left_mask.sum(), right_mask.sum()
    DLK, DRK = optimized_number_squared_for_lbl(y_left), optimized_number_squared_for_lbl(y_right)

    # Gini index calculation
    return DR * (DL**2- DLK) + DL * (DR**2 - DRK)


def find_best_split(data, num_features):
    X, y = data[:, :-1], data[:, -1]
    best_feature_idx, best_threshold = 0, 0
    best_metric = float('inf')
    for feature_idx in range(num_features):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            y_left, y_right = y[left_mask], y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            DL, DR = left_mask.sum(), right_mask.sum()
            DLK, DRK = number_squared_for_lbl(y_left), number_squared_for_lbl(y_right)

            # g0 = best_DL* best_DR*(DR*(DL**2-DLK)+DL*(DR**2-DRK))
            # gini_index = DL*DR*(DR*(DL**2-DLK)+DL*(DR**2-DRK))

            gini_index = DL * (DL - DLK) + DL * (DR - DRK)
            # g0 = DR*(-DLK)+DL*(-DRK)
            # gb = best_DR*(- best_DLK)+best_DL*(-best_DRK)

            if gini_index < best_metric:
                best_metric = gini_index
                best_feature_idx, best_threshold = feature_idx, threshold
    return best_feature_idx, best_threshold


def split_dataset(X_column, threshold):
    """Split the dataset based on the given feature and threshold"""
    left_idx = np.argwhere(X_column <= threshold).flatten()

    right_idx = np.argwhere(X_column > threshold).flatten()
    return left_idx, right_idx


def predict(tree, sample):
    """Predict the label for a given sample using the decision tree"""
    if tree.is_leaf_node():
        return tree.value
    if sample[tree.feature_index] <= tree.threshold:
        return predict(tree.left, sample)
    return predict(tree.right, sample)


# 测试样例：


# file_path = 'sepsis_data.csv'
# sepsis_data = pd.read_csv(file_path)
# X = sepsis_data.drop("id", axis=1).drop("y", axis=1).to_numpy()
# y = sepsis_data['y'].to_numpy()
# data_length = len(X)
# print(data_length)
# # Load Iris dataset
# iris = load_iris()

# # data_length = len(X)
# iris = load_iris()
#iris = load_breast_cancer()
# iris =  load_wine()
# iris = load_digits()

# import numpy as np
# from ucimlrepo import fetch_ucirepo
# raisin = fetch_ucirepo(id=20)
# # #327
# # # data (as pandas dataframes)
# X = raisin.data.features
# y = raisin.data.targets
# # Numerical data imputation
# X = X.dropna()  # Removes rows with any missing values in X
# y = y.loc[X.index]  # Ensure y matches the cleaned X rows
# # # iris  = load_digits()
# from sklearn.preprocessing import LabelEncoder
#
# # Create a label encoder object
# le = LabelEncoder()
#
# # Apply label encoding to the column that contains 'Besni' (let's say it's the first column for this example)
# X= X.apply(le.fit_transform)
# # X, y = iris.data, iris.target
# y = le.fit_transform(y)
# iris = load_digits()
# X, y = iris.data, iris.target
#
# from ucimlrepo import fetch_ucirepo
#
# # fetch dataset
# skin_segmentation = fetch_ucirepo(id=229)
#
# # data (as pandas dataframes)
# X = skin_segmentation.data.features
# y = skin_segmentation.data.targets
from ucimlrepo import fetch_ucirepo

# fetch dataset
covertype = fetch_ucirepo(id=31)
X = covertype.data.features
y = covertype.data.targets
X = X.to_numpy()
y = y.to_numpy()
num_bins =2
# binner = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy="quantile")
# X= binner.fit_transform(X)
#
#
#
# # 使用 VarianceThreshold 移除常量特征
# selector = VarianceThreshold(threshold=0)  # 0 表示移除方差为 0 的特征，即常量特征
# X_reduced = selector.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
data = np.hstack((X_train, y_train.reshape(-1, 1)))
# Build the Decision Tree
max_depth = 4
start = time.time()
tree = build_tree(data, 4)
end = time.time()

# Predictions
predictions = [predict(tree, sample) for sample in X_test]
accuracy = accuracy_score(y_test, predictions)
print("Our Accuracy:", accuracy)
#
# start = time.time()
# tree = build_tree(data, max_depth=10)
# end = time.time()
#
# # Predictions
# predictions = [predict(tree, sample) for sample in X_test]
# accuracy = accuracy_score(y_test, predictions)
# print("Our Accuracy-8:", accuracy)
#
# start = time.time()
# tree = build_tree(data, max_depth=15)
# end = time.time()

# # Predictions
# predictions = [predict(tree, sample) for sample in X_test]
# accuracy = accuracy_score(y_test, predictions)
# print("Our Accuracy-16:", accuracy)

# X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=4, criterion="gini")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("CART Accuracy:", accuracy)

clf = DecisionTreeClassifier(max_depth=8, criterion="gini")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("CART Accuracy-8:", accuracy)


clf = DecisionTreeClassifier(max_depth=16, criterion="gini")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("CART Accuracy-16:", accuracy)

clf = DecisionTreeClassifier(max_depth=4, criterion="entropy")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("ID3 Accuracy-4:", accuracy)


clf = DecisionTreeClassifier(max_depth=8, criterion="entropy")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("ID3 Accuracy-8:", accuracy)

clf = DecisionTreeClassifier(max_depth=16, criterion="entropy")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("ID3 Accuracy-16:", accuracy)

clf = DecisionTreeClassifier(max_depth=4, criterion="log_loss")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("C4.5 Accuracy-4:", accuracy)

clf = DecisionTreeClassifier(max_depth=8, criterion="log_loss")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("C4.5 Accuracy:", accuracy)
clf = DecisionTreeClassifier(max_depth=16, criterion="log_loss")

# Train the Classifier
start = time.time()
clf.fit(X_train, y_train)
end = time.time()

# Make Predictions
y_pred = clf.predict(X_test)


# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("C4.5 Accuracy:", accuracy)









# import matplotlib.pyplot as plt
# import networkx as nx

#
# def draw_decision_tree(node, pos=None, level=0, width=100, loc=0, G=None):
#     if G is None:
#         G = nx.DiGraph()
#         pos = {loc: (0, 0)}
#
#     # 为当前节点添加标签
#     label = f"{node.feature_index}\n{node.threshold}" if not node.is_leaf_node() else node.value
#     G.add_node(loc, label=label)
#
#     if node.left:
#         G.add_edge(loc, 2 * loc + 1)
#         pos[2 * loc + 1] = (pos[loc][0] - width / 2 ** level, pos[loc][1] - 1)
#         draw_decision_tree(node.left, pos, level + 1, width, 2 * loc + 1, G)
#
#     if node.right:
#         G.add_edge(loc, 2 * loc + 2)
#         pos[2 * loc + 2] = (pos[loc][0] + width / 2 ** level, pos[loc][1] - 1)
#         draw_decision_tree(node.right, pos, level + 1, width, 2 * loc + 2, G)
#
#     return G, pos
#
#
# def our_plot_tree(node):
#     G, pos = draw_decision_tree(node)
#     labels = {node: G.nodes[node]['label'] for node in G.nodes}
#     nx.draw(G, pos, labels=labels, with_labels=True, node_size=1000, node_color="skyblue", font_size=10)
#
#
# # 第一棵树的绘制
# plt.figure(figsize=(20, 10))
# our_plot_tree(tree)  # 这里假设our_plot_tree是一个自定义函数
# plt.title("Decision Tree Visualization 1")
#
# # 第二棵树的绘制
# plt.figure(figsize=(20, 10))
# plot_tree(clf)  # 这里假设clf是一个决策树模型
# plt.title("Decision Tree Visualization 2")
#
# # 显示图形
# plt.show()