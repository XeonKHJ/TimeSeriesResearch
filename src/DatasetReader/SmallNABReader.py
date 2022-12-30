import os
import os.path
import pandas
import torch
from DatasetReader.DatasetReader import IDatasetReader

class SmallNABReader(IDatasetReader):
    def __init__(self, folderPath) -> None:
        super().__init__()
        self.folderPath = folderPath

    def read(self):
        fileList = os.listdir(self.folderPath)
        fulldata = list()
        dataTimestampLengths = list()
        featureSize = 1
        maxDataLength = 0
        for file in fileList:
            filePath = os.path.join(self.folderPath, file)
            data = pandas.read_csv(filePath)
            datasetItem = data.value.to_list()
            datasetItem = datasetItem[0:100]
            fulldata.append(datasetItem)
            maxDataLength = max(datasetItem.__len__(), maxDataLength)
        fulldata.sort(key=(lambda elem:len(elem)), reverse=True)
        dataTensor = torch.zeros([fulldata.__len__(), maxDataLength, featureSize])
        for i in range(fulldata.__len__()):
            dataTensor[i][0:fulldata[i].__len__()] = torch.tensor(fulldata[i][:]).reshape([-1,1])
            dataTimestampLengths.append(fulldata[i].__len__())
        return dataTensor, dataTimestampLengths