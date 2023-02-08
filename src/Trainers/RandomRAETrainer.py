from Trainers.ITrainer import ITrainer
import torch
import time
import os.path as path
import numpy

class RandomRAETrainer(ITrainer):
    def __init__(self, mlModel, logger, taskName, tsLambda=0.1):
        self.aeModel = mlModel
        self.lossFunc = torch.nn.MSELoss()
        self.ts = torch.zeros([0])
        self.tsLambda = tsLambda
        self.optimizer = torch.optim.Adam(self.aeModel.parameters(), lr=1e-3)
        self.step2Optimizer = None
        self.raeTsPath = "SavedModels/random_raels.pt"
        self.modelFolderPath = "SavedModels"
        self.epoch = 0
        self.logger = logger
        self.taskName = taskName

    def train(self, trainSet, trainSetLength, labelSet):
        self.epoch += 1
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
                    self.ts = torch.rand(trainSet.shape, requires_grad=True, device=torch.device('cuda'))
                else:
                    self.ts = torch.rand(trainSet.shape, requires_grad=True)
            self.step2Optimizer = torch.optim.Adam([self.ts], lr=1e-3)
        noise = torch.tensor(numpy.random.normal(0, 1, (trainSet.shape[0], trainSet.shape[1], trainSet.shape[2])), dtype=torch.float32, device=torch.device('cuda'))
        startTime = time.perf_counter()
        tl = trainSet - self.ts
        tlInput = torch.cat((tl, noise), 2).detach()
        x = self.aeModel(tlInput, trainSetLength)
        fowardTime = time.perf_counter() - startTime
        loss1 = self.lossFunc(tl, x)
        loss2 = self.lossFunc(x, trainSet)
        tsLoss = self.tsLambda * (1/(torch.norm(self.ts, p=1)))
        loss = loss1 + 0.1 * loss2 + tsLoss
        loss.backward()
        self.optimizer.step()
        self.step2Optimizer.step()
        self.optimizer.zero_grad()
        self.step2Optimizer.zero_grad()
        

        backwardTime = time.perf_counter() - startTime
        # loss = loss1 + tsLoss
        print("epoch\t",self.epoch,"\tloss\t", format(loss.item(), ".7f"), "\t", format(loss1.item(), ".7f"), "\t", format(loss2.item(), ".7f"), "\t", format(tsLoss.item() / self.tsLambda, ".7f"), "\tts\t", format(torch.norm(self.ts, p=1).item(), ".7f"), "\tfoward\t", format(fowardTime, ".3f"), "\tbackward\t", format(backwardTime, ".3f"))
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.aeModel.eval()
        for abnormalIdx in range(len(validsetLengths)):
            abnormalInputSet, abnormalLabelSet, abnormalLengths = self.aeModel.getInputTensor(
                validDataset, validsetLengths)
            noise = torch.tensor(numpy.random.normal(0, 1, (abnormalInputSet.shape[0], abnormalInputSet.shape[1], abnormalInputSet.shape[2])), dtype=torch.float32, device=torch.device('cuda'))
            input = torch.cat((abnormalInputSet, noise), 2)
            abnormalOutput = self.aeModel(input, abnormalLengths)    
            tl = abnormalOutput[abnormalIdx]
            t = abnormalLabelSet[abnormalIdx]
            ts = t - tl
            tlList = tl.reshape([-1]).tolist()
            tList = t.reshape([-1]).tolist()
            tsList = ts.abs().reshape([-1]).tolist()
            # selftsList = self.ts[0].reshape([-1]).tolist()
            maxDiff = (torch.abs(abnormalLabelSet - abnormalOutput)).max().item()
            print("max diff\t", maxDiff)
            self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.taskName+ '-raetrainer-' + storeName + "-" + str(abnormalIdx))

    def save(self, filename=None):
        pass
        # torch.save(self.ts, self.raeTsPath)
        torch.save(self.aeModel.state_dict(), path.join(self.modelFolderPath, self.taskName + ".pt"))