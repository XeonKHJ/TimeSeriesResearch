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
        self.step2Optimizer = None

    def train(self, trainSet, trainSetLength, labelSet):
        if self.ts.shape == torch.Size([0]):
            self.ts = torch.rand(trainSet.shape, requires_grad=True)
            self.step2Optimizer = torch.optim.Adam([self.ts], lr=1e-2)
        tl = trainSet - self.ts
        x = self.mlModel(tl, trainSetLength)
        loss = self.lossFunc(tl, x) + self.tsLambda * torch.norm(self.ts)
        loss.backward()
        self.optimizer.step()
        self.step2Optimizer.step()
        self.optimizer.zero_grad()
        self.step2Optimizer.zero_grad()
        lambdaTs1 = self.tsLambda * torch.sum(torch.abs(self.ts))
        # lambdaTs1.backward()

        print("λ||tl||\t",lambdaTs1)

        return loss
        