from DataNormalizer.IDataNormalizer import IDataNormalizer
import torch
import sys

class StandardScalerDataNormalizer(IDataNormalizer):
    def __init__(self):
        self.datas = []
        self.lengths = []
        return

    def addDatasetToRef(self, dataset, lengths):
        self.datas.append(dataset)
        self.lengths.append(lengths)
        return

    def normalizeDataset(self, dataset, lengths):
        beginDataTensor = None
        for dataIdx in range(len(self.datas)):
            d = self.datas[dataIdx]
            l = self.lengths[dataIdx]
            for di in range(len(l)):
                lens = l.int().tolist()
                curD = d[di, 0:lens[di]]
                if beginDataTensor == None:
                    beginDataTensor = curD[:]
                else:
                    beginDataTensor = torch.cat((beginDataTensor, curD), 0)
        mean = torch.mean(beginDataTensor)
        std = torch.std(beginDataTensor)
        dataset = (dataset - mean) / std
        return dataset