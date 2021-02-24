import numpy as np


class RootNode:
    def __init__(self, data, feature, target, start):
        self.__tree = list()
        self.__start_point = start
        self.__feature = feature
        self.__target = target
        self.__dataset = data

    def __bin_split(self, i):
        thresh = np.mean(self.__dataset[self.__feature].iloc[i:i + self.__start_point])
        lSet = self.__dataset[self.__dataset[self.__feature] < thresh]
        rSet = self.__dataset[self.__dataset[self.__feature] >= thresh]
        return lSet, rSet, thresh

    def __calculate_RMSE(self, l, r):
        l_avg = np.mean(l[self.__target])
        r_avg = np.mean(r[self.__target])
        RMSE = np.sum(np.sqrt((l[self.__target] - l_avg) ** 2)) + np.sum(np.sqrt((r[self.__target] - r_avg) ** 2))
        return RMSE

    def best_split(self):
        for i in range(0, self.__dataset.shape[0] - self.__start_point):
            left, right, thresh = self.__bin_split(i)
            rmse = self.__calculate_RMSE(left, right)
            self.__tree.append([thresh, rmse])
        return [self.__tree, ]
