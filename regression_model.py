class RootNode:
    def __init__(self, data, feature, start):
        self.tree = dict()
        self.start_point = start
        self.feature = feature
        self.dataset = data.sort_values(by=feature)

    def bin_split(self):
        thresh = self.dataset[self.feature].iloc[:self.start_point]
        lSet = self.dataset[self.dataset[self.feature] < thresh]
        rSet = self.dataset[self.dataset[self.feature] >= thresh]

