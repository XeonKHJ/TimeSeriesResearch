from DataNormalizer.IDataNormalizer import IDataNormalizer
import torch
import sys

class PerDataNormalizer(IDataNormalizer):
    def __init__(self, perData = False):
        self.max = None
        self.min = None
        self.perData = perData
        pass

    def addDatasetToRef(self, dataset, lengths):
        curMax = torch.max(dataset, 1)[0]
        curMin = torch.min(dataset, 1)[0]
        if self.max == None:
            self.max = curMax
        else:
            self.max = torch.max(self.max, curMax)
        
        if self.min == None:
            self.min = curMin
        else:
            self.min = torch.min(self.min, curMin)
        return

    def normalizeDataset(self, dataset, lengths):
        lengths = lengths.int().tolist()
        for dataIdx in range(len(lengths)):
            curMax = torch.max(dataset[dataIdx, 0:lengths[dataIdx]], 0)[0]
            curMax = curMax.repeat(1, lengths[dataIdx]).reshape([lengths[dataIdx], 1])
            curMin = torch.min(dataset[dataIdx, 0:lengths[dataIdx]], 0 )[0]
            curMin = curMin.repeat(1, lengths[dataIdx]).reshape([lengths[dataIdx], 1])
            dataset[dataIdx, 0:lengths[dataIdx]] = (dataset[dataIdx, 0:lengths[dataIdx]] - curMin) / (curMax - curMin)
        return dataset