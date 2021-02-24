import numpy as np


def bin_split_df(data, feature, value):
    mat0 = data.iloc[np.nonzero(data[feature].values > value)[0],:].iloc[0]
    mat1 = data.iloc[np.nonzero(data[feature].values <= value)[0], :].iloc[0]
    return mat0, mat1

def create_tree():


class TreeNode:
    def __init__(self, feature, value, left, right):
        self.feature_to_split = feature
        self.value_of_split = value
        self.left_branch = left
        self.right_branch = right
