import os
import os.path
import pandas
from DatasetReader.DatasetReader import DatasetReader

class NABReader(DatasetReader):
    def __init__(self, folderPath) -> None:
        super().__init__()
        self.folderPath = folderPath

    def read(self):
        fileList = os.listdir(self.folderPath)
        for file in fileList:
            filePath = os.path.join(self.folderPath, file)
            data = pandas.read_csv(filePath)
            print(data)