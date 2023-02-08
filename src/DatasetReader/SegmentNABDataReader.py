import pandas
import torch
import datetime, re, json, os.path as path
from DatasetReader.DatasetReader import IDatasetReader

class SegmentNABDataReader(IDatasetReader):
    def __init__(self, filePath, windowSize = 100, step=1) -> None:
        super().__init__()
        self.filePath = filePath
        self.fileList = [filePath]
        self.windowSize = windowSize
        self.step = step
        self.labelPath = '../../NAB/labels/combined_labels.json'

    def read(self):
        label = self.readLabels()
        fileList = self.fileList
        fulldata = list()
        dataTimestampLengths = list()
        featureSize = 1
        maxDataLength = 0
        for file in fileList:
            filePath = file
            data = pandas.read_csv(filePath)
            datasetItem = data.value.to_list()
            timestamps = data['timestamp'].tolist()
            for idx in range(len(timestamps)):
                datetimes = re.split('[- :]',timestamps[idx])
                datetimes = datetime.datetime(int(datetimes[0]),int(datetimes[1]),int(datetimes[2]),int(datetimes[3]),int(datetimes[4]),int(datetimes[5]))
                timestamps[idx] = datetimes
            fulldata.append({'set':datasetItem, 'timestamps': timestamps, 'filename':path.basename(file)})
        
        fulldata = self.segement(fulldata)
        for data in fulldata:
            maxDataLength = max(len(data['set']), maxDataLength)
        fulldata.sort(key=(lambda elem:len(elem)), reverse=True)
        
        dataTensor = torch.zeros([fulldata.__len__(), maxDataLength, featureSize])
        labelTensor = torch.ones([fulldata.__len__(), maxDataLength, featureSize])
        for i in range(fulldata.__len__()):
            dataTensor[i][0:fulldata[i]['set'].__len__()] = torch.tensor(fulldata[i]['set'][:]).reshape([-1,1])
            for outlierTimeStamp in label[fulldata[i]['filename']]:
                try:
                    outlierIdx = fulldata[i]['timestamps'].index(outlierTimeStamp)
                    labelTensor[i][outlierIdx] = 0
                except:
                    pass
            dataTimestampLengths.append(fulldata[i]['set'].__len__())

        if torch.cuda.is_available():
            return dataTensor.cuda(), dataTimestampLengths, dataTensor.cuda(), labelTensor.cuda()
        else:
            return dataTensor, dataTimestampLengths, dataTensor, labelTensor
    
    def segement(self, fulldata):
        segementedData = list()
        for data in fulldata:
            totalSteps = len(data['set']) - self.windowSize + 1
            for step in range(totalSteps):
                curItem = {
                    'set': data['set'][step:step+self.windowSize],
                    'timestamps': data['timestamps'][step:step+self.windowSize],
                    'filename': data['filename']
                }
                segementedData.append(curItem)
        return segementedData

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