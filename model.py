import numpy as np


def bin_split_df(data, feature, value):
    mat0 = data.iloc[np.nonzero(data[feature].values > value)[0], :].iloc[0]
    mat1 = data.iloc[np.nonzero(data[feature].values <= value)[0], :].iloc[0]
    return mat0, mat1


def create_tree(data, leaf_type=reg_leaf, error_type=reg_error, option=(1, 4)):
    feature, value = choose_best_split(data, leaf_type, error_type, option)

    if feature is None:
        return value

    retTree = {'spInd': feature,
               'spVal': value}
    lSet,rSet = bin_split_df(data,feature,value)
    retTree['left'] = create_tree(lSet,leaf_type,error_type,option)
    retTree['right'] = create_tree(rSet,leaf_type,error_type,option)

    return retTree


class TreeNode:
    def __init__(self, feature, value, left, right):
        self.feature_to_split = feature
        self.value_of_split = value
        self.left_branch = left
        self.right_branch = right
