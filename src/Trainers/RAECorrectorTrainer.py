from Network.LstmAutoencoder import LstmAutoencoder
from Trainers.CorrectorTrainer import CorrectorTrainer
from Trainers.ITrainer import ITrainer
import torch
import time

from Trainers.RAETrainer import RAETrainer

class RAECorrectorTrainer(ITrainer):
    def __init__(self, aeModel, correctorModel, logger):
        self.aeModel = aeModel
        self.raeTrainer = RAETrainer(aeModel, logger, 'raecorrectorae')
        self.correctorTrainer = CorrectorTrainer(aeModel, correctorModel ,logger)
        self.epoch = 0
        self.isAeTrained = False

    def setAbnormal(self, abnormalSet, abnormalSetLengths):
        if torch.cuda.is_available():
            self.abnormalSet = abnormalSet.cuda()
        else:
            self.abnormalSet = abnormalSet
        self.abnormalSetLengths = abnormalSetLengths

    def train(self, trainSet, trainSetLength, labelSet):  
        aeEpochPerRound = 1000
        correctorEpochPerRound = 1
        if not self.isAeTrained:
            for i in range(aeEpochPerRound):
                aeLoss = self.raeTrainer.train(trainSet, trainSetLength, labelSet)
            self.isAeTrained = True
        # for i in range(correctorEpochPerRound):
        correctorLoss = self.correctorTrainer.train(self.abnormalSet, self.abnormalSetLengths, None)
        loss = correctorLoss
        print('\tcorrectorLoss\t', correctorLoss.item())
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.raeTrainer.evalResult(validDataset, validsetLengths, storeName)
        self.correctorTrainer.evalResult(validDataset, validsetLengths, 'RAECorrectorTrainer')

    def save(self):
        self.raeTrainer.save()
        self.correctorTrainer.save('correctortrainer')