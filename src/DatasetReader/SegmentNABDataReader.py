import os
import os.path
import pandas
import torch
from DatasetReader.DatasetReader import IDatasetReader

class SegmentNABDataReader(IDatasetReader):
    def __init__(self, filePath, windowSize = 100, step=1) -> None:
        super().__init__()
        self.fileList = [filePath]
        self.windowSize = windowSize
        self.step = step

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
        
        fulldata = self.segement(fulldata)
        for data in fulldata:
            maxDataLength = max(data.__len__(), maxDataLength)
        fulldata.sort(key=(lambda elem:len(elem)), reverse=True)
        
        dataTensor = torch.zeros([fulldata.__len__(), maxDataLength, featureSize])
        for i in range(fulldata.__len__()):
            dataTensor[i][0:fulldata[i].__len__()] = torch.tensor(fulldata[i][:]).reshape([-1,1])
            dataTimestampLengths.append(fulldata[i].__len__())

        return dataTensor, dataTimestampLengths, dataTensor
    
    def segement(self, fulldata):
        segementedData = list()
        for data in fulldata:
            totalSteps = len(data) - self.windowSize + 1
            for step in range(totalSteps):
                curData = data[step:step+self.windowSize]
                segementedData.append(curData)
        return segementedData