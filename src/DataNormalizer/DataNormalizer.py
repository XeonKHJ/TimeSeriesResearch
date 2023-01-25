from DataNormalizer.IDataNormalizer import IDataNormalizer
import torch
import sys

class DataNormalizer(IDataNormalizer):
    def __init__(self):
        self.max = None
        self.min = None
        pass

    def addDatasetToRef(self, dataset):
        curMax = torch.max(torch.max(dataset, 1)[0], 0)[0]
        curMin = torch.min(torch.min(dataset, 1)[0], 0)[0]
        if self.max == None:
            self.max = curMax
        else:
            self.max = torch.max(self.max, curMax)
        
        if self.min == None:
            self.min = curMin
        else:
            self.min = torch.min(self.min, curMin)
        return

    def normalizeDataset(self, dataset):
        curMin = self.min.reshape([1,-1 ,self.min.shape[0]]).expand_as(dataset)
        curMax = self.max.reshape([1,-1 ,self.max.shape[0]]).expand_as(dataset)
        dataset = (dataset - curMin) / (curMax - curMin)
        return dataset