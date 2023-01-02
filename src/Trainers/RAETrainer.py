from Trainers.ITrainer import ITrainer
import torch

class RAETrainer(ITrainer):
    def __init__(self, mlModel):
        self.mlModel = mlModel
        self.lossFunc = self.lossFunc = torch.nn.MSELoss()
        self.ts = 0
        self.tsLambda = 0.0001
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=1e-2, weight_decay=0.001)

    def train(self, trainSet, trainSetLength, labelSet):
        self.tl = trainSet - self.ts

        output = self.mlModel(trainSet, trainSetLength)
        loss = self.lossFunc(output, self.tl)
        self.optimizer.backward()

        lambdaTs1 = self.tsLambda * torch.sum(torch.abs(self.ts))
        loss = loss + lambdaTs1

        self.tl = self.mlModel(self.tl, trainSetLength)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
        