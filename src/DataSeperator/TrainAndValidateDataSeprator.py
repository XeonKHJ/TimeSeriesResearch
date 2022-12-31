from DataSeperator.DataSeperator import IDataSeperator

class NoSepDataSeperator(IDataSeperator):
    def __init__(self, percentage):
        self.percentage = percentage

    def getValidationSet(self, dataset):
        len = dataset.__len__()
        sepIdx = len * self.percentage
        return dataset[0:sepIdx+1]
        
    def getTrainningSet(self, dataset):
        len = dataset.__len__()
        sepIdx = len * self.percentage
        return dataset[sepIdx+1:dataset.__len__()]