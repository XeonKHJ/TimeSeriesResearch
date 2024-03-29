import os
import os.path
import pandas
import torch
from DatasetReader.DatasetReader import IDatasetReader

class HSSReader(IDatasetReader):
    def __init__(self, folderPath, isNormal=True) -> None:
        super().__init__()
        self.folderPath = folderPath
        self.isNormal = isNormal

    def read(self):
        fileList = os.listdir(self.folderPath)
        fulldata = list()
        dataTimestampLengths = list()
        featureSize = 20
        maxDataLength = 0
        for file in fileList:
            if self.isNormal:
                toRead = file.__contains__('normal')
            else:
                toRead = file.__contains__('anomalous')
            if toRead:
                columnData = list()
                filePath = os.path.join(self.folderPath, file)
                data = pandas.read_csv(filePath)
                columns = data.columns.tolist()
                for column in columns:
                    columnData.append(data[column].to_list())
                fulldata.append(columnData)
                maxDataLength = max(len(columnData[0]), maxDataLength)
        fulldata.sort(key=(lambda elem:len(elem[0])), reverse=True)
        dataTensor = torch.zeros([fulldata.__len__(), maxDataLength, featureSize])
        # dataTensor = torch.tensor(fulldata)
        for i in range(fulldata.__len__()):
            dataTensor[i, 0:len(fulldata[i][0])] = torch.tensor(fulldata[i]).transpose(1,0)
            dataTimestampLengths.append(len(fulldata[i][0]))
        labelTensor = dataTensor[:,:,1:2]
        dataTensor = dataTensor[:,:,2:dataTensor.shape[2]]
        return dataTensor, dataTimestampLengths, labelTensor