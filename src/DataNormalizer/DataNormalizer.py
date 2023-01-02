from DataNormalizer.IDataNormalizer import IDataNormalizer
import torch
import sys

class DataNormalizer(IDataNormalizer):
    def __init__(self):
        self.max = sys.float_info.min
        self.min = sys.float_info.max
        pass

    def addDatasetToRef(self, dataset):
        self.max = max(self.max, torch.max(dataset))
        self.min = min(self.min, torch.min(dataset))
        return

    def normalizeDataset(self, dataset):
        dataset = (dataset - self.min) / (self.max - self.min)
        return dataset