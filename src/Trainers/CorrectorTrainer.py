from Trainers.ITrainer import ITrainer
import torch
import time

class CorrectorTrainer(ITrainer):
    def __init__(self, mlModel, correctorModel, logger):
        self.mlModel = mlModel
        self.correctorModel = correctorModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.correctorOptimizer = torch.optim.Adam(self.correctorModel.parameters(), lr=1e-3)
        self.logger = logger
        self.epoch = 0

    def train(self, trainSet, trainSetLength, labelSet):
        self.mlModel.eval()
        tl = self.mlModel(trainSet, trainSetLength)
        ts = trainSet - tl
        tr = self.correctorModel(trainSet, trainSetLength)
        loss = self.lossFunc(tl, tr)
        loss.backward()
        self.correctorOptimizer.step()
        self.correctorOptimizer.zero_grad()
        plotIdx = 5
        if self.epoch % 100  == 0:
            self.logger.logResults([
                trainSet[plotIdx].reshape([-1]).tolist(),
                tr[plotIdx].reshape([-1]).tolist()
            ], ['abdata', 'tr'])
        self.epoch += 1
        return loss
        