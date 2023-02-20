from Trainers.ITrainer import ITrainer
import torch
import os.path as path


class CorrectorTrainer(ITrainer):
    def __init__(self, aeModel, correctorModel, logger):
        self.aeModel = aeModel
        self.correctorModel = correctorModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.mlModelOptimizer = torch.optim.Adam(
            self.aeModel.parameters(), lr=1e-3)
        self.correctorOptimizer = torch.optim.Adam(
            self.correctorModel.parameters(), lr=1e-3)
        self.logger = logger
        self.epoch = 0
        self.threadhold = 0.07
        self.modelFolderPath = "SavedModels"

    def train(self, trainSet, trainSetLength, labelSet):
        self.aeModel.eval()
        self.correctorModel.train()

        tl = self.aeModel(trainSet, trainSetLength).detach()
        ts = torch.abs(trainSet - tl)
        tsMask = (ts >= self.threadhold)

        correctorInputTensor = torch.cat((tl, ts), 2)
        tr1 = self.correctorModel(correctorInputTensor, trainSetLength)
        self.maskedTls = tr1.masked_fill(tsMask, 0)
        loss1 = self.lossFunc(self.maskedTls, trainSet.masked_fill(tsMask, 0))
        loss1.backward()
        self.correctorOptimizer.step()
        self.correctorOptimizer.zero_grad()

        tr2 = self.correctorModel(correctorInputTensor, trainSetLength)
        # tr3 = trainSet.masked_fill(tsMask, tr2[masked_fill])
        tr3 = torch.where(ts >= self.threadhold, tr2, trainSet)
        aetr = self.aeModel(tr3, trainSetLength).detach()
        loss2 = self.lossFunc(tr3, aetr)
        loss2.backward()
        self.correctorOptimizer.step()
        self.correctorOptimizer.zero_grad()

        loss = loss1+loss2
        print("corrector\tloss1\t", format(loss1.item(), ".7f"),
              "\tloss2\t", format(loss2.item(), ".7f"), "\tloss\t", loss.item())

        # For evaluation
        self.storeResult(trainSet, tr1, tr2, tr3, tl, aetr)
        return loss

    def storeResult(self, trainSet, tr1, tr2, tr3, tl, aetr):
        self.trainSet = trainSet
        self.tr1 = tr1
        self.tr2 = tr2
        self.tr3 = tr3
        self.tl = tl
        self.aetr = aetr


    def evalResult(self, validDataset, validsetLengths, storeName=None):
        for i in range(self.trainSet.shape[0]):
            self.logger.logResults([
                self.trainSet[i].reshape([-1]).tolist(),
                self.tr1[i].reshape([-1]).tolist(),
                self.tl[i].reshape([-1]).tolist(),
                self.aetr[i].reshape([-1]).tolist()
            ], ['abdata', 'tr', 'tl', 'aetr'], str(i) + '-res1')
            self.logger.logResults([
                self.maskedTls[i].reshape([-1]).tolist(),
                self.trainSet[i].reshape([-1]).tolist()
            ], ['masked', 'og'], str(i) + '-res2')

    def save(self, filename=None):
        torch.save(self.correctorModel.state_dict(), path.join(self.modelFolderPath, filename + ".pt"))
