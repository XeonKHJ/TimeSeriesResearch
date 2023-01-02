from Trainers.ITrainer import ITrainer
import torch

class Trainer(ITrainer):
    def __init__(self, model):
        self.mlModel = model
        self.lossFunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=1e-2)

    def train(self, trainSet, trainSetLength, labelSet):
        output = self.mlModel(trainSet, trainSetLength)
        loss = self.lossFunc(output, labelSet)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss