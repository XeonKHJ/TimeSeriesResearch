from DataSeperator.DataSeperator import IDataSeperator

class NoSepDataSeperator(IDataSeperator):
    def __init__(self):
        return

    def getValidationSet(self, dataset, lengths):
        return dataset[0:dataset.__len__()], lengths[0:dataset.__len__()]
        
    def getTrainningSet(self, dataset, lengths):
        return dataset[0:dataset.__len__()], lengths[0:dataset.__len__()]