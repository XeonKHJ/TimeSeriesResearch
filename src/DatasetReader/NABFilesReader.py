import os
import os.path
import pandas
import torch
import json
import datetime
import re
import os.path as path
from DatasetReader.DatasetReader import IDatasetReader

class NABFilesReader(IDatasetReader):
    def __init__(self, repoPath, dataType, filePrefix):
        super().__init__()
        self.repoPath = repoPath
        self.dataFolders = list()
        self.labelPath = os.path.join(repoPath, 'labels', 'combined_windows.json')
        self.filePrefix = filePrefix
        if dataType == 'artificial':
            self.dataFolders.append(os.path.join(repoPath, 'data', 'artificialNoAnomaly'))
            self.dataFolders.append(os.path.join(repoPath, 'data', 'artificialWithAnomaly'))
        else:
            self.dataFolders.append(os.path.join(repoPath, 'data', dataType))

    def read(self):
        # label = self.readLabels()
        windowLabels = self.readWindowLabels()
        fileList = list()
        for folder in self.dataFolders:
            curFileList = os.listdir(folder)
            for file in curFileList:
                if file.__contains__(self.filePrefix):
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
        labelTensor = torch.zeros([fulldata.__len__(), maxDataLength, featureSize])
        for i in range(fulldata.__len__()):
            dataTensor[i][0:fulldata[i]['set'].__len__()] = torch.tensor(fulldata[i]['set'][:]).reshape([-1,1])
            for outlierTimeStamp in windowLabels[fulldata[i]['filename']]:
                try:
                    outlierIdx = fulldata[i]['timestamps'].index(outlierTimeStamp[0])
                    endOutlierIdx = fulldata[i]['timestamps'].index(outlierTimeStamp[1])
                    labelTensor[i][outlierIdx:endOutlierIdx+1] = 1
                except:
                    pass
            dataTimestampLengths.append(fulldata[i]['set'].__len__())
        dataTimestampLengths = torch.tensor(dataTimestampLengths)
        # dataTensor = torch.cat((dataTensor, labelTensor), 2)
        if torch.cuda.is_available():
            return dataTensor.cuda(), dataTimestampLengths.cuda(), labelTensor.cuda(), fileList
        else:
            return dataTensor, dataTimestampLengths, labelTensor, fileList

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

    def readWindowLabels(self):
        labels = json.load(open(self.labelPath))
        newLabels = {}
        for label in labels:
            datas = labels[label]
            outlierTimeStamps = []
            for i in range(len(datas)):
                beginDatetime = re.split('[- : .]',datas[i][0])
                endDatetime = re.split('[- : .]',datas[i][1])
                beginDatetime = datetime.datetime(int(beginDatetime[0]),int(beginDatetime[1]),int(beginDatetime[2]),int(beginDatetime[3]),int(beginDatetime[4]),int(beginDatetime[5]))
                endDatetime = datetime.datetime(int(endDatetime[0]),int(endDatetime[1]),int(endDatetime[2]),int(endDatetime[3]),int(endDatetime[4]),int(endDatetime[5]))
                outlierTimeStamps.append([beginDatetime,endDatetime])
            newLabel = path.basename(label)
            newLabels[newLabel] = outlierTimeStamps

        return newLabels