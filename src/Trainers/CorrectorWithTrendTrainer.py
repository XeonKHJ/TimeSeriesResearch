from Trainers.ITrainer import ITrainer
import torch
import os.path as path


class CorrectorWithTrendTrainer(ITrainer):
    def __init__(self, generatorModel, trendModel, correctorModel, logger, lambda1 = 1, lambda2 = 0.01):
        self.generatorModel = generatorModel
        self.trendModel = trendModel
        self.correctorModel = correctorModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.mlModelOptimizer = torch.optim.Adam(
            self.generatorModel.parameters(), lr=1e-3)
        self.correctorOptimizer = torch.optim.Adam(
            self.correctorModel.parameters(), lr=1e-3)
        self.logger = logger
        self.epoch = 0
        self.threadhold = 0.07
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.modelFolderPath = "SavedModels"
        self.tl = None
        self.trendTl = None

    def train(self, trainSet, trainSetLength, labelSet):
        self.generatorModel.train()
        self.trendModel.train()
        self.correctorModel.train()

        if self.tl == None:
            self.tl = self.generatorModel(trainSet, trainSetLength).detach()
            self.trend = self.trendModel(trainSet, trainSetLength).detach()
            self.ts = torch.abs(trainSet - self.tl)
            self.tsMask = (self.ts >= self.threadhold)

        correctorInputTensor = torch.cat((self.tl, self.ts), 2)
        tr = self.correctorModel(correctorInputTensor, trainSetLength)

        self.maskedTls = tr.masked_fill(self.tsMask, 0)
        nonErrorLoss = self.lossFunc(self.maskedTls, trainSet.masked_fill(self.tsMask, 0))

        # maskedTr = torch.where(self.ts >= self.threadhold, tr, trainSet)
        maskedTr = tr
        aetr = self.generatorModel(maskedTr, trainSetLength).detach()
        restoreLoss = self.lossFunc(maskedTr, aetr)

        trendTr = self.trendModel(tr, trainSetLength).detach()
        self.trendTr = trendTr
        trendLoss = self.lossFunc(tr, trendTr)

        self.ts = trainSetLength - tr

        loss = nonErrorLoss + self.lambda1 * restoreLoss + self.lambda2 * trendLoss
    
        loss.backward()
        self.correctorOptimizer.step()
        self.correctorOptimizer.zero_grad()

        print("non-error loss\t", nonErrorLoss.item(), "\trestore loss\t", restoreLoss.item(), "\ttrend loss\t", trendLoss.item())

        # For evaluation
        self.storeResult(trainSet, tr)
        return loss

    def storeResult(self, trainSet, tr):
        self.trainSet = trainSet
        self.tr = tr


    def evalResult(self, validDataset, validsetLengths, storeName=None):
        for i in range(self.trainSet.shape[0]):
            self.logger.logResults([
                self.trainSet[i].reshape([-1]).tolist(),
                self.tl[i].reshape([-1]).tolist(),
                self.tr[i].reshape([-1]).tolist(),
                self.trendTr[i].reshape([-1]).tolist(),
            ], ['abdata', 'tl', 'tr', 'trend'], str(i) + '-res1')

    def save(self, filename=None):
        torch.save(self.correctorModel.state_dict(), path.join(self.modelFolderPath, filename))
        torch.save(self.ts, path.join(self.modelFolderPath, "SavedModels/recoverTs.pt"))
