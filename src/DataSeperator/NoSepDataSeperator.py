from DataSeperator.DataSeperator import IDataSeperator

class NoSepDataSeperator(IDataSeperator):
    def __init__(self):
        pass

    def getValidationSet(self, dataset):
        return dataset[0:dataset.__len__()]
        
    def getTrainningSEt(self, dataset):
        pass