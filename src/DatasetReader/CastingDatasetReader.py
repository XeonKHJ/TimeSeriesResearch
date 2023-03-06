import os
import os.path
import pandas
import torch
import json
import datetime
import re
import os.path as path
from DatasetReader.DatasetReader import IDatasetReader

class CastingDatasetReader(IDatasetReader):
    def __init__(self, datasetPath):
        super().__init__()
        self.datasetPath = datasetPath
        self.contextName = 'context.csv'

    def read(self):
        contexts = pandas.read_csv(os.path.join(self.datasetPath, self.contextName))
        datas = list()
        filenames = datas['filename'].tolist()
        dataTimestampLengths = list()
        for context in contexts['filenames']:
            curFileName = context.filename
            fileData = pandas.read_csv(path.join(self.datasetPath, curFileName))
            datas.append(curFileName)

        dataTensor = torch.tensor(datas)
        if torch.cuda.is_available():
            return dataTensor.cuda(), dataTimestampLengths.cuda(), labelTensor.cuda(), fileList
        else:
            return dataTensor, dataTimestampLengths, labelTensor, fileList