import numpy as np
import pandas as pd


class RegressionTree:
    def __init__(self, x, y, data, interval, max_depth=0):
        self.x = x
        self.y = y
        self.data = data
        self.interval = interval
        self.depth = max_depth


class BestSplit(RegressionTree):
    def __init__(self, x, y, data, interval, max_depth):
        super().__init__(x, y, data, interval, max_depth)
        self.tree = list()

    def bin_split(self, i, feat):
        thresh = np.mean(self.data[feat].iloc[i:i + self.interval])
        lSet = self.data[self.data[feat] < thresh]
        rSet = self.data[self.data[feat] >= thresh]
        return lSet, rSet, thresh

    def calculate_RMSE(self, l, r):
        l_avg = np.mean(l[self.y])
        r_avg = np.mean(r[self.y])
        RMSE = np.sum(np.sqrt((l[self.y] - l_avg) ** 2)) + np.sum(
            np.sqrt((r[self.y] - r_avg) ** 2))
        return RMSE

    def best_split(self):
        for feat in self.x:
            for i in range(0, self.data.shape[0] - self.interval):
                left, right, thresh = self.bin_split(i, feat)
                rmse = self.calculate_RMSE(left, right)
                self.tree.append([feat, thresh, rmse])
        self.tree = pd.DataFrame(self.tree, columns=['feature', 'thresh', 'cost'])
        best_root = self.tree[self.tree['cost'] == np.min(self.tree['cost'])]

        if best_root.shape[0] > 1:
            best_root = best_root.sort_values(by='thresh').head(1)
        return [best_root.feature.values[0], best_root.thresh.values[0], self.tree]


class TreeBuilder(BestSplit):
    def __init__(self, x, y, data, interval, max_depth):
        super().__init__(x, y, data, interval, max_depth)

    def bin_split(self, data, **kwargs):
        feature, node = BestSplit(data=data, x=self.x, y=self.y, interval=self.interval, max_depth=5).best_split()[:2]
        left = data[data[feature] < node]
        right = data[data[feature] >= node]

        return left, right, node, feature

    def builder(self, data, depth=0):
        lSet, rSet, node, feature = self.bin_split(data, )
        Tree = {'node': node, 'feature': feature}

        if depth < self.depth and lSet.shape[0] >= 20:
            Tree['left'] = np.mean(lSet[self.y])
            Tree['right'] = self.builder(rSet, depth + 1)
        else:
            Tree['right'] = np.mean(rSet[self.y])

        return Tree

