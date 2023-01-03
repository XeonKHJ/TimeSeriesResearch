from Trainers.ITrainer import ITrainer
import torch

class RAETrainer(ITrainer):
    def __init__(self, mlModel):
        self.mlModel = mlModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.ts = torch.zeros([0])
        self.tsLambda = 0.00001
        self.l = 10
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=1e-2, weight_decay=0.001)

    def train(self, trainSet, trainSetLength, labelSet):
        if self.ts.shape == torch.Size([0]):
            self.ts = torch.rand(trainSet.shape, requires_grad=True)
        tl = trainSet - self.ts
        x = self.mlModel(tl, trainSetLength)
        loss = self.lossFunc(tl, x) + self.tsLambda * torch.norm(self.ts)
        loss.backward()
        grad = self.ts.grad
        print("grad\t", grad)

        z = self.ts - (1/self.l) * grad
        compareMap1 = (self.tsLambda / self.l < z).long()
        ts1 = z - compareMap1 * (self.tsLambda/self.l)
        compareMap2 = (-self.tsLambda / self.l > z).long()
        ts2 = z + compareMap2 * (self.tsLambda/self.l)
        compareMap3 = (~(torch.abs(z) <= self.tsLambda / self.l)).long()
        ts3 = z * compareMap3
        self.ts = ts1 * (compareMap1) + ts2 * (compareMap2) + ts3 * (compareMap3)
        self.ts = self.ts.detach()
        self.ts.requires_grad_(True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        lambdaTs1 = self.tsLambda * torch.sum(torch.abs(self.ts))
        # lambdaTs1.backward()

        print("λ||tl||\t",lambdaTs1)

        return loss
        