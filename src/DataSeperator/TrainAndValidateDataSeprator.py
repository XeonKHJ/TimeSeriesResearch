from DataSeperator.DataSeperator import IDataSeperator

class TrainAndValidateDataSeprator(IDataSeperator):
    def __init__(self, percentage):
        self.percentage = percentage

    def getValidationSet(self, dataset, dataLengths):
        len = dataset.__len__()
        sepIdx = int(len * self.percentage)
        return dataset[0:sepIdx+1,:,:], dataLengths[0:sepIdx+1]
        
    def getTrainningSet(self, dataset, dataLengths):
        len = dataset.__len__()
        sepIdx = int(len * self.percentage)
        return dataset[sepIdx+1:dataset.__len__(),:,:], dataLengths[sepIdx+1:dataset.__len__()]