from Trainers.ITrainer import ITrainer
import torch
import time

class CorrectorTrainer(ITrainer):
    def __init__(self, mlModel, correctorModel, logger):
        self.mlModel = mlModel
        self.correctorModel = correctorModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.mlModelOptimizer = torch.optim.Adam(self.mlModel.parameters(), lr=1e-3)
        self.correctorOptimizer = torch.optim.Adam(self.correctorModel.parameters(), lr=1e-3)
        self.logger = logger
        self.epoch = 0

    def train(self, trainSet, trainSetLength, labelSet):
        self.mlModel.eval()
        self.correctorModel.train()

        tl = self.mlModel(trainSet, trainSetLength).detach()
        ts = torch.abs(trainSet - tl)
        tsMask = (ts >= 0.07)

        tls = torch.cat((tl, ts), 2)   
        tr = self.correctorModel(tls, trainSetLength)
        maskedTls = tr.masked_fill(tsMask, 0)
        loss1 = self.lossFunc(maskedTls, trainSet.masked_fill(tsMask, 0))
        loss1.backward()
        self.correctorOptimizer.step()
        self.correctorOptimizer.zero_grad()
        tr2 = self.correctorModel(tls, trainSetLength)
        aetr = self.mlModel(tr2, trainSetLength).detach()
        loss2 = self.lossFunc(tr2, aetr)
        loss2.backward()
        self.correctorOptimizer.step()
        self.correctorOptimizer.zero_grad()
        loss = loss1+loss2
        plotIdx = 5
        print("corrector\tloss1\t", format(loss1.item(), ".7f"), "\tloss2\t", format(loss2.item(), ".7f"), "\tloss\t", loss.item())
        # if self.epoch % 100  == 0:
        #     self.logger.logResults([
        #         trainSet[plotIdx].reshape([-1]).tolist(),
        #         tr[plotIdx].reshape([-1]).tolist(),
        #         tl[plotIdx].reshape([-1]).tolist(),
        #         aetr[plotIdx].reshape([-1]).tolist()
        #     ], ['abdata', 'tr', 'tl', 'aetr'], 'res2')
        #     self.logger.logResults([
        #         maskedTls[plotIdx].reshape([-1]).tolist(),
        #         trainSet[plotIdx].reshape([-1]).tolist()
        #     ], ['masked', 'og'], 'res1')
        if self.epoch % 1000 == 0:
            for i in range(trainSet.shape[0]):
                self.logger.logResults([
                    trainSet[i].reshape([-1]).tolist(),
                    tr[i].reshape([-1]).tolist(),
                    tl[i].reshape([-1]).tolist(),
                    aetr[i].reshape([-1]).tolist()
                ], ['abdata', 'tr', 'tl', 'aetr'], str(i) + '-res1')
                self.logger.logResults([
                    maskedTls[i].reshape([-1]).tolist(),
                    trainSet[i].reshape([-1]).tolist()
                ], ['masked', 'og'], str(i) +'-res2')
        self.epoch += 1
        return loss1 + loss2
        