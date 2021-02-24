class TreeNode():
    def __init__(self, feature, value, left, right):
        self.feature_to_split = feature
        self.value_of_split = value
        self.left_branch = left
        self.right_branch = right
