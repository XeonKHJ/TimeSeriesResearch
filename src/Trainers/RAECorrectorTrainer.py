from Network.LstmAutoencoder import LstmAutoencoder
from Trainers.CorrectorTrainer import CorrectorTrainer
from Trainers.ITrainer import ITrainer
import torch
import time

from Trainers.RAETrainer import RAETrainer

class RAECorrectorTrainer(ITrainer):
    def __init__(self, aeModel, correctorModel, logger, abnormalSet, abnormalSetLengths):
        self.aeModel = aeModel
        self.raeTrainer = RAETrainer(aeModel, logger)
        self.correctorTrainer = CorrectorTrainer(aeModel, correctorModel ,logger)
        self.abnormalSet = abnormalSet
        self.abnormalSetLengths = abnormalSetLengths
        self.epoch = 0

    def train(self, trainSet, trainSetLength, labelSet):
        aeEpochPerRound = 1
        correctorEpochPerRound = 1
        if self.epoch % aeEpochPerRound:
            aeLoss = self.raeTrainer.train(trainSet, trainSetLength, labelSet)
        if self.epoch % correctorEpochPerRound:
            correctorLoss = self.correctorTrainer.train(self.abnormalSet, self.abnormalSetLengths, None)
        loss = aeLoss + correctorLoss
        print('RAECorrectorTrainer\tloss\t',loss.item(), '\taeLoss\t', aeLoss.item(), '\tcorrectorLoss\t', correctorLoss.item())
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.raeTrainer.evalResult(validDataset, validsetLengths)
        self.correctorTrainer.evalResult(validDataset, validsetLengths)