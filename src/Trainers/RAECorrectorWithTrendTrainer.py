from Network.LstmAutoencoder import LstmAutoencoder
from Trainers.CorrectorTrainer import CorrectorTrainer
from Trainers.ITrainer import ITrainer
import torch
import time

from Trainers.RAETrainer import RAETrainer

class RAECorrectorWithTrendTrainer(ITrainer):
    def __init__(self, generatorModel, correctorModel, trendModel, logger):
        self.generatorModel = generatorModel
        self.trendModel = trendModel
        self.raeTrainer = RAETrainer(generatorModel, logger, 'raecorrectorae')
        self.correctorTrainer = CorrectorTrainer(generatorModel, correctorModel ,logger)
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
        correctorEpochPerRound = 1
        if not self.isAeTrained:
            for i in range(aeEpochPerRound):
                _ = self.raeTrainer.train(trainSet, trainSetLength, labelSet)
            self.isAeTrained = True
        if not self.isTrendTrained:
            for i in range(aeEpochPerRound):
                _ = self.trendModel.train(trainSet, trainSetLength, labelSet)
        correctorLoss = self.correctorTrainer.train(self.abnormalSet, self.abnormalSetLengths, None)
        loss = correctorLoss
        print('\tcorrectorLoss\t', correctorLoss.item())
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.raeTrainer.evalResult(validDataset, validsetLengths, "RAECorrectorTrainerGeneratorModel")
        self.trendModel.evalResult(validDataset, validsetLengths, "RAECorrectorTrainerTrendModel")
        self.correctorTrainer.evalResult(validDataset, validsetLengths, 'RAECorrectorTrainerWithTrend')

    def save(self):
        self.raeTrainer.save()
        self.correctorTrainer.save('correctortrainer')