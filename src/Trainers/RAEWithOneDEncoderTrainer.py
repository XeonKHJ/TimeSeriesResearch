from Trainers.ITrainer import ITrainer
import torch
import time
import os.path as path

class RAETrainer(ITrainer):
    def __init__(self, mlModel, logger, taskName):
        self.aeModel = mlModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.ts = torch.zeros([0])
        self.tsLambda = 0.01
        self.optimizer = torch.optim.Adam(self.aeModel.parameters(), lr=1e-3)
        self.step2Optimizer = None
        self.raeTsPath = "SavedModels/raels.pt"
        self.modelFolderPath = "SavedModels"
        self.epoch = 0
        self.logger = logger
        self.taskName = taskName

    def train(self, trainSet, trainSetLength, labelSet):
        self.aeModel.train()
        if self.ts.shape == torch.Size([0]):
            try:
                if torch.cuda.is_available():
                    print('CUDA is avaliable')
                    self.ts = torch.load(self.raeTsPath, map_location = torch.device('cuda'))
                else:
                    print('CUDA is not avaliable.')
                    self.ts = torch.load(self.raeTsPath)
            except:
                if torch.cuda.is_available():
                    self.ts = torch.zeros(trainSet.shape, requires_grad=True, device=torch.device('cuda'))
                else:
                    self.ts = torch.zeros(trainSet.shape, requires_grad=True)
            self.step2Optimizer = torch.optim.Adam([self.ts], lr=1e-3)
        startTime = time.perf_counter()
        tl = trainSet - self.ts
        x = self.aeModel(tl, trainSetLength)
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
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.aeModel.eval()
        for abnormalIdx in range(len(validsetLengths)):
            abnormalInputSet, abnormalLabelSet, abnormalLengths = self.aeModel.getInputTensor(
                validDataset, validsetLengths)
            abnormalOutput = self.aeModel(abnormalInputSet, abnormalLengths)    
            tl = abnormalOutput[abnormalIdx]
            t = abnormalLabelSet[abnormalIdx]
            ts = t - tl
            tlList = tl.reshape([-1]).tolist()
            tList = t.reshape([-1]).tolist()
            tsList = ts.reshape([-1]).tolist()
            # selftsList = self.ts[0].reshape([-1]).tolist()
            maxDiff = (torch.abs(abnormalLabelSet - abnormalOutput)).max().item()
            print("max diff\t", maxDiff)
            self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], 'raetrainer-' + storeName + "-" + str(abnormalIdx))

    def save(self, filename=None):
        torch.save(self.ts, self.raeTsPath)
        torch.save(self.aeModel.state_dict(), path.join(self.modelFolderPath, self.taskName + ".pt"))