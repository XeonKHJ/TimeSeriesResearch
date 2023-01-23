from Network.LstmAutoencoder import LstmAutoencoder
from Trainers.CorrectorTrainer import CorrectorTrainer
from Trainers.CorrectorWithTrendTrainer import CorrectorWithTrendTrainer
from Trainers.ITrainer import ITrainer
import torch
import time

from Trainers.RAETrainer import RAETrainer
from Trainers.Trainer import Trainer

class RAECorrectorWithTrendTrainer(ITrainer):
    def __init__(self, generatorModel, correctorModel, trendModel, logger):
        self.generatorModel = generatorModel
        self.raeTrainer = RAETrainer(generatorModel, logger, 'raecorrectorae')
        self.correctorTrainer = CorrectorWithTrendTrainer(generatorModel, trendModel, correctorModel ,logger)
        self.trendTrainer = Trainer(trendModel, 'RAECorrectorWithTrendTrainer', logger, 1e-3)
        self.epoch = 0
        self.isAeTrained = False
        self.isTrendTrained = False

    def setAbnormal(self, abnormalSet, abnormalSetLengths):
        if torch.cuda.is_available():
            self.abnormalSet = abnormalSet.cuda()
        else:
            self.abnormalSet = abnormalSet
        self.abnormalSetLengths = abnormalSetLengths

    def train(self, trainSet, trainSetLength, labelSet):  
        aeEpochPerRound = 1000
        if not self.isAeTrained:
            for i in range(aeEpochPerRound):
                _ = self.raeTrainer.train(trainSet, trainSetLength, labelSet)
            self.isAeTrained = True
        if not self.isTrendTrained:
            for i in range(aeEpochPerRound):
                _ = self.trendTrainer.train(trainSet, trainSetLength, labelSet)
            self.isTrendTrained = True
        correctorLoss = self.correctorTrainer.train(self.abnormalSet, self.abnormalSetLengths, None)
        loss = correctorLoss
        print('\tcorrectorLoss\t', correctorLoss.item())
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.raeTrainer.evalResult(validDataset, validsetLengths, "RAECorrectorTrainer-GeneratorModel")
        self.trendTrainer.evalResult(validDataset, validsetLengths, "RAECorrectorTrainer-TrendModel")
        self.correctorTrainer.evalResult(validDataset, validsetLengths, 'RAECorrectorTrainer-WithTrend')

    def save(self):
        self.correctorTrainer.save('RAECorrectorWithTrendTrainer')