import torch

class ITrainer(object):
    def __init__(self):
        pass
    def train(self, trainSet, labelSet):
        pass
    def save(filename=None):
        pass
    
    def setAbnormal(self, abnormalSet, abnormalSetLengths):
        if torch.cuda.is_available():
            self.abnormalSet = abnormalSet.cuda()
        else:
            self.abnormalSet = abnormalSet
        self.abnormalSetLengths = abnormalSetLengths