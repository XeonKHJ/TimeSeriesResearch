from DataNormalizer.IDataNormalizer import IDataNormalizer
import torch
import sys

class StandardScalerDataNormalizer(IDataNormalizer):
    def __init__(self):
        self.datas = []
        return

    def addDatasetToRef(self, dataset):
        self.datas.append(dataset)
        return

    def normalizeDataset(self, dataset):
        beginDataTensor = None
        for dataset in self.datas:
            if beginDataTensor == None:
                beginDataTensor = dataset[:]
            else:
                beginDataTensor = torch.cat((beginDataTensor, dataset), 0)
        mean = torch.mean(beginDataTensor)
        std = torch.std(beginDataTensor)
        dataset = (dataset - mean) / std
        return dataset