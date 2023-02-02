from Trainers.ITrainer import ITrainer
import time
import torch
import os.path as path
import numpy

class TadGanTrainer(ITrainer):
    def __init__(self, generator, discriminator, taskName, logger, learningRate=1e-3):
        self.generator = generator
        self.discriminator = discriminator
        self.lossFunc = torch.nn.MSELoss()
        self.generatorOptimizer = torch.optim.Adam(self.generator.parameters(), lr=learningRate)
        self.discriminatorOptimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learningRate)
        self.taskName = taskName
        self.logger = logger
        self.modelFolderPath = "SavedModels"

    def train(self, trainSet, trainSetLength, labelSet):
        self.generator.train()
        self.discriminator.train()

        valid = torch.ones(trainSet.shape, requires_grad=False)
        fake = torch.zeros(trainSet.shape, requires_grad=False)

        startTime = time.perf_counter()
        # generate fake sample
        self.generatorOptimizer.zero_grad()
        z = torch.tensor(numpy.random.normal(0, 1, (trainSet.shape[0], trainSet[1], trainSet[2])))
        generatedSamples = self.generator(z)
        loss = self.lossFunc(self.discriminator(generatedSamples), valid)
        loss.backward()
        self.generatorOptimizer.step()
        updateGeneratorTime = time.perf_counter()

        # train discriminator
        self.discriminatorOptimizer.zero_grad()
        discriminateReal = self.discriminator(trainSet)
        discriminateFake = self.discriminator(generatedSamples.detach())
        realLoss = self.lossFunc(discriminateReal, valid)
        fakeLoss = self.lossFunc(discriminateFake, fake)
        discriminatorLoss = (realLoss + fakeLoss) / 2
        discriminatorLoss.backward()
        discriminatorLoss.step()
        updateDiscriminatorTime = time.perf_counter()

        print("\tloss\t", loss.item() , "\tgtime\t", updateGeneratorTime-startTime, "\tdtime\t", updateDiscriminatorTime - updateGeneratorTime)
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.generator.eval()
        z = torch.tensor(numpy.random.normal(0, 1, (validDataset.shape[0], validDataset[1], validDataset[2])))
        for abnormalIdx in range(len(validsetLengths)):
            abnormalInputSet, abnormalLabelSet, abnormalLengths = self.mlModel.getInputTensor(
                validDataset, validsetLengths)
            abnormalOutput = self.generator(abnormalInputSet, abnormalLengths) 
            for featIdx in range(validDataset.shape[2]):
                tl = abnormalOutput[abnormalIdx,:,featIdx]
                t = abnormalLabelSet[abnormalIdx,:,featIdx]
                ts = t - tl
                tlList = tl.reshape([-1]).tolist()
                tList = t.reshape([-1]).tolist()
                tsList = ts.reshape([-1]).tolist()
                maxDiff = (torch.abs(abnormalLabelSet - abnormalOutput)).max().item()
                # print("max diff\t", maxDiff)
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.taskName + '-' + storeName + "-idx" + str(abnormalIdx) + '-feat' + str(featIdx))

    def save(self, filename=None):
        if filename == None:
            filename = self.taskName + ".pt"
        else:
            filename = self.taskName + '-' + filename + ".pt"
        torch.save(self.mlModel.state_dict(), path.join(self.modelFolderPath, filename))