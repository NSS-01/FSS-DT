import torch
from sklearn.metrics import accuracy_score

from NssMPC.crypto.primitives import ArithmeticSecretSharing
from NssMPC.common.ring.ring_tensor import RingTensor
class Tree:
    def __init__(self, f=-1, t=-1):
        self.f = f
        self.t = t

    def __str__(self):
        return f"f={self.f}, t={self.t}"
class DecisionTreeNode:
    """A decision tree node class for binary tree"""

    def __init__(self, feature_index=None, threshold=None):
        """
        - feature_index: Index of the feature used for splitting.
        - threshold: Threshold value for splitting.
        - left: Left subtree.
        - right: Right subtree.
        - value: Value of the node if it's a leaf node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = None
        self.right = None
        self.value = None


    def is_leaf_node(self):
        """Check if the node is a leaf node"""
        return self.value is not None

    def __str__(self):
        return  f"feature_index, threshold:{(self.feature_index,self.threshold)}"
def predict(tree, sample):
    """Predict the label for a given sample using the decision tree"""
    if tree.is_leaf_node():
        return tree.value
    if sample[tree.feature_index] <= tree.threshold:
        return predict(tree.left, sample)
    return predict(tree.right, sample)

def Merge_trees(server_tree_model, cleint_tree_model):
    print(len(server_tree_model),len(cleint_tree_model))
    if len(server_tree_model)!=len(cleint_tree_model):
        return None
    def build_node(index):
        if cleint_tree_model[index].m==-2 and server_tree_model[index].m ==-2:
            node = DecisionTreeNode()
            node.value = (cleint_tree_model[index].t+server_tree_model[index].t).tensor
            return node
        node  =  DecisionTreeNode()
        node.feature_index = cleint_tree_model[index].m+server_tree_model[index].m+1
        node.threshold = cleint_tree_model[index].t+server_tree_model[index].t+1
        node.left = build_node(2*index+1)
        node.right = build_node(2*index+2)
        return node
    return build_node(0)
if __name__ =="__main__":
    data_name = 'covertype'
    # data_name = 'skin'
    server_model = torch.load(f"../model/{data_name}_server_model.pth")
    client_model = torch.load(f"../model/{data_name}_client_model.pth")
    print(len(server_model), len(client_model))
    # for c,s in zip(server_model,client_model):
    #     print(f"client: m:{c.m} t:{c.t}\n server m:{s.m}, t:{s.t}")
    #     print(f"m: {c.m+s.m+1}, t: {c.t+s.t+1}")
    tree = Merge_trees(server_model, client_model)
    X_train, X_test, y_train, y_test = torch.load(f'../data/{data_name}.pth')

    predictions = [predict(tree, sample) for sample in X_test]
    accuracy = accuracy_score(y_test, predictions)
    print("Our Accuracy:", accuracy)



