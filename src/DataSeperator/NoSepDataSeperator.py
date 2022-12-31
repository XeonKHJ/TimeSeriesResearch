from DataSeperator.DataSeperator import IDataSeperator

class NoSepDataSeperator(IDataSeperator):
    def __init__(self):
        return

    def getValidationSet(self, dataset):
        return dataset[0:dataset.__len__()]
        
    def getTrainningSet(self, dataset):
        return dataset[0:dataset.__len__()]