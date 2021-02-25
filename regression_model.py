import numpy as np
import pandas as pd


class RootNode:
    def __init__(self, data, feature, target, start):
        self.tree = list()
        self.start_point = start
        self.feature = feature
        self.target = target
        self.dataset = data

    def __bin_split(self, i):
        thresh = np.mean(self.dataset[self.feature].iloc[i:i + self.start_point])
        lSet = self.dataset[self.dataset[self.feature] < thresh]
        rSet = self.dataset[self.dataset[self.feature] >= thresh]
        return lSet, rSet, thresh

    def __calculate_RMSE(self, l, r):
        l_avg = np.mean(l[self.target])
        r_avg = np.mean(r[self.target])
        RMSE = np.sum(np.sqrt((l[self.target] - l_avg) ** 2)) + np.sum(np.sqrt((r[self.target] - r_avg) ** 2))
        return RMSE

    def best_split(self):
        for i in range(0, self.dataset.shape[0] - self.start_point):
            left, right, thresh = self.__bin_split(i)
            rmse = self.__calculate_RMSE(left, right)
            self.tree.append([thresh, rmse])
        self.tree = pd.DataFrame(self.tree, columns=['thresh', 'cost'])
        best_root = self.tree[self.tree['cost'] == np.min(self.tree['cost'])].thresh.values[0]
        return [self.tree, best_root]


class TreeBuilder:
    def __init__(self, data, feature, target, start, break_point):
        self.data = data
        self.feature = feature
        self.target = target
        self.start = start
        self.break_point = break_point

    def bin_split(self, data):
        node = RootNode(data, self.feature, self.target, self.start).best_split()[1]
        left = data[data[self.feature] < node]
        right = data[data[self.feature] >= node]

        return left, right, node

    def builder(self, data):
        lSet, rSet, node = self.bin_split(data)
        Tree = {'node': node}

        if lSet.shape[0] >= self.break_point:
            Tree['left'] = self.builder(lSet)
        else:
            Tree['left'] = np.mean(lSet[self.target])

        if rSet.shape[0] >= self.break_point:
            Tree['right'] = self.builder(rSet)
        else:
            Tree['right'] = np.mean(rSet[self.target])

        return Tree
