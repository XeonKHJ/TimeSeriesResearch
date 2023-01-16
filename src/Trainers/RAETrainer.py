from Trainers.ITrainer import ITrainer
import torch
import time

class RAETrainer(ITrainer):
    def __init__(self, mlModel, logger):
        self.mlModel = mlModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.ts = torch.zeros([0])
        self.tsLambda = 0.001
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=1e-3)
        self.step2Optimizer = None
        self.modelPath = "SavedModels/raels.pt"
        self.epoch = 0
        self.logger = logger

    def train(self, trainSet, trainSetLength, labelSet):
        if self.ts.shape == torch.Size([0]):
            try:
                if torch.cuda.is_available():
                    self.ts = torch.load(self.modelPath, map_location = torch.device('cuda'))
                else:
                    self.ts = torch.load(self.modelPath)
            except:
                if torch.cuda.is_available():
                    self.ts = torch.zeros(trainSet.shape, requires_grad=True, device=torch.device('cuda'))
                else:
                    self.ts = torch.zeros(trainSet.shape, requires_grad=True)
            self.step2Optimizer = torch.optim.Adam([self.ts], lr=1e-3)
        startTime = time.perf_counter()
        tl = trainSet - self.ts
        x = self.mlModel(tl, trainSetLength)
        fowardTime = time.perf_counter() - startTime
        loss1 = self.lossFunc(tl, x)
        loss1.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss2 = self.tsLambda * torch.norm(self.ts, p=1)
        loss2.backward()
        self.step2Optimizer.step()
        self.step2Optimizer.zero_grad()
        backwardTime = time.perf_counter() - startTime
        loss = loss1 + loss2
        print("loss\t", format(loss.item(), ".7f"), "\t", format(loss1.item(), ".7f"), "\t", format(loss2.item(), ".7f"), "\tfoward\t", format(fowardTime, ".3f"), "\tbackward\t", format(backwardTime, ".3f"))
        if self.epoch % 100 == 0:
            # tslist = self.ts[0].reshape([-1]).tolist()
            # self.logger.logSingleResult(tslist, "ts")
            torch.save(self.ts, self.modelPath)
        self.epoch += 1

        return loss
        