import os
import os.path
import pandas
import torch
import json
import datetime
import re
import os.path as path
from DatasetReader.DatasetReader import IDatasetReader

class NABFoldersReader(IDatasetReader):
    def __init__(self, folderPath) -> None:
        super().__init__()
        self.folderPath = folderPath
        self.labelPath = '../../NAB/labels/combined_labels.json'

    def read(self):
        label = self.readLabels()
        fileList = list()
        for folder in self.folderPath:
            curFileList = os.listdir(folder)
            for file in curFileList:
                fileList.append(os.path.join(folder, file))
        fulldata = list()
        dataTimestampLengths = list()
        featureSize = 1
        maxDataLength = 0
        rawData = {}
        datetimeList = {}
        for file in fileList:
            filePath = file
            data = pandas.read_csv(filePath)
            datasetItem = data.value.to_list()
            timestamps = data['timestamp'].tolist()
            for idx in range(len(timestamps)):
                datetimes = re.split('[- :]',timestamps[idx])
                datetimes = datetime.datetime(int(datetimes[0]),int(datetimes[1]),int(datetimes[2]),int(datetimes[3]),int(datetimes[4]),int(datetimes[5]))
                timestamps[idx] = datetimes
            fulldata.append({'set':datasetItem, 'filename': os.path.basename(file), 'timestamps':timestamps})
            maxDataLength = max(datasetItem.__len__(), maxDataLength)
            rawData[file] = data
        fulldata.sort(key=(lambda elem:len(elem['set'])), reverse=True)
        dataTensor = torch.zeros([fulldata.__len__(), maxDataLength, featureSize])
        labelTensor = torch.ones([fulldata.__len__(), maxDataLength, featureSize])
        for i in range(fulldata.__len__()):
            dataTensor[i][0:fulldata[i]['set'].__len__()] = torch.tensor(fulldata[i]['set'][:]).reshape([-1,1])
            for outlierTimeStamp in label[fulldata[i]['filename']]:
                outlierIdx = fulldata[i]['timestamps'].index(outlierTimeStamp)
                labelTensor[i][outlierIdx] = 0
            dataTimestampLengths.append(fulldata[i]['set'].__len__())

        if torch.cuda.is_available():
            return dataTensor.cuda(), dataTimestampLengths, dataTensor.cuda(), labelTensor.cuda(), fileList
        else:
            return dataTensor, dataTimestampLengths, dataTensor, labelTensor, fileList
    
    def readLabels(self):
        labels = json.load(open(self.labelPath))
        newLabels = {}
        for label in labels:
            datas = labels[label]
            outlierTimeStamps = []
            for i in range(len(datas)):
                datetimes = re.split('[- :]',datas[i])
                datetimes = datetime.datetime(int(datetimes[0]),int(datetimes[1]),int(datetimes[2]),int(datetimes[3]),int(datetimes[4]),int(datetimes[5]))
                outlierTimeStamps.append(datetimes)
            newLabel = path.basename(label)
            newLabels[newLabel] = outlierTimeStamps

        return newLabels