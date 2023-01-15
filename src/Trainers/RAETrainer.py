from Trainers.ITrainer import ITrainer
import torch
import time

class RAETrainer(ITrainer):
    def __init__(self, mlModel):
        self.mlModel = mlModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.ts = torch.zeros([0])
        self.tsLambda = 0.001
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=1e-2, weight_decay=0.001)
        self.step2Optimizer = None

    def train(self, trainSet, trainSetLength, labelSet):
        if self.ts.shape == torch.Size([0]):
            if torch.cuda.is_available():
                self.ts = torch.rand(trainSet.shape, requires_grad=True, device=torch.device('cuda'))
            else:
                self.ts = torch.rand(trainSet.shape, requires_grad=True)
            self.step2Optimizer = torch.optim.Adam([self.ts], lr=1e-2)
        startTime = time.perf_counter()
        tl = trainSet - self.ts
        x = self.mlModel(tl, trainSetLength)
        fowardTime = time.perf_counter() - startTime
        loss1 = self.lossFunc(tl, x)
        loss2 = self.tsLambda * torch.norm(self.ts, p=1)
        loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()
        self.step2Optimizer.step()
        self.optimizer.zero_grad()
        self.step2Optimizer.zero_grad()
        backwardTime = time.perf_counter() - startTime
        print("loss\t", format(loss.item(), ".7f"), "\t", format(loss1.item(), ".7f"), "\t", format(loss2.item(), ".7f"), "\tfoward\t", format(fowardTime, ".3f"), "\tbackward\t", format(backwardTime, ".3f"))

        return loss
        