from DataNormalizer.IDataNormalizer import IDataNormalizer
import torch
import sys

class NegPosDataNormalizer(IDataNormalizer):
    def __init__(self):
        self.max = None
        self.min = None
        pass

    def addDatasetToRef(self, dataset):
        if self.max == None:
            self.max = torch.max(dataset)
        else:
            self.max = max(self.max, torch.max(dataset))
        
        if self.min == None:
            self.min = torch.min(dataset)
        else:
            self.min = min(self.min, torch.min(dataset))
        return

    def normalizeDataset(self, dataset):
        dataset = (dataset - self.min) / (self.max - self.min) - 0.5
        return dataset