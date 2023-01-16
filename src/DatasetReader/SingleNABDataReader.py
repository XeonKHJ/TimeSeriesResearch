import os
import os.path
import pandas
import torch
from DatasetReader.DatasetReader import IDatasetReader

class SingleNABDataReader(IDatasetReader):
    def __init__(self, filePath) -> None:
        super().__init__()
        self.fileList = [filePath]

    def read(self):
        fileList = self.fileList
        fulldata = list()
        dataTimestampLengths = list()
        featureSize = 1
        maxDataLength = 0
        for file in fileList:
            filePath = file
            data = pandas.read_csv(filePath)
            datasetItem = data.value.to_list()
            fulldata.append(datasetItem)
            maxDataLength = max(datasetItem.__len__(), maxDataLength)
        fulldata.sort(key=(lambda elem:len(elem)), reverse=True)
        dataTensor = torch.zeros([fulldata.__len__(), maxDataLength, featureSize])
        for i in range(fulldata.__len__()):
            dataTensor[i][0:fulldata[i].__len__()] = torch.tensor(fulldata[i][:]).reshape([-1,1])
            dataTimestampLengths.append(fulldata[i].__len__())
        return dataTensor, dataTimestampLengths