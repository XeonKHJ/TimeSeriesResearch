from Trainers.ITrainer import ITrainer
import torch
import time
import os.path as path
import numpy

class TimeGANTrianer(ITrainer):
    def __init__(self, generator, discriminator, staticFeatureModel, tempFeatureModel, logger, taskName, tsLambda=0.1):
        self.staticFeatureModel = staticFeatureModel
        self.tempFeatureModel = tempFeatureModel
        self.discriminator = discriminator
        self.generator = generator
        self.lossFunc = torch.nn.MSELoss()
        self.generatorOptim = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        self.disciriminatorOptim = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        self.staticFeatureOptim = torch.optim.Adam(self.staticFeatureModel.parameters(), lr=1e-3)
        self.tempFeatureOptim = torch.optim.Adam(self.tempFeatureModel.parameters(), lr=1e-3) 
        self.modelFolderPath = "SavedModels"
        self.epoch = 0
        self.logger = logger
        self.taskName = taskName
        self.isCuda = torch.cuda.is_available()

    def train(self, trainSet, trainSetLength, labelSet):
        self.discriminator.train()
        self.generator.train()
        self.staticFeatureModel.train()
        self.tempFeatureModel.train()
        if self.isCuda:
            zt = torch.tensor(numpy.random.normal(0, 1, (trainSet.shape[0], trainSet.shape[1], trainSet.shape[2])), dtype=torch.float32, device=torch.device('cuda'), requires_grad=False)
            postives = torch.ones(trainSet.shape, device=torch.device('cuda'))
            negs = torch.zeros(trainSet.shape, device=torch.device('cuda'))
        else:
            zt = torch.tensor(numpy.random.normal(0, 1, (trainSet.shape[0], trainSet.shape[1], trainSet.shape[2])), dtype=torch.float32, device=torch.device('cpu'), requires_grad=False)
        startTime = time.perf_counter()

        # train generator
        self.generatorOptim.zero_grad()
        self.staticFeatureOptim.zero_grad()
        self.tempFeatureOptim.zero_grad()
        staticFeature = self.staticFeatureModel(trainSet, trainSetLength)
        staticFeature = staticFeature.repeat(1, trainSet.shape[1], 1)
        tempFeature = self.tempFeatureModel(trainSet, trainSetLength)
        tlInput = torch.cat((zt, staticFeature, tempFeature), 2)
        tempFeature = self.generator(tlInput, trainSetLength)
        generatorLoss = self.lossFunc(self.discriminator(tempFeature, trainSetLength), postives)
        generatorLoss.backward()
        self.generatorOptim.step()
        self.staticFeatureOptim.step()
        self.tempFeatureOptim.step()

        # train 
        self.disciriminatorOptim.zero_grad()
        realResult = self.discriminator(trainSet, trainSetLength)
        fakeResult = self.discriminator(tempFeature.detach(), trainSetLength)
        realLoss = self.lossFunc(realResult, postives)
        fakeLoss = self.lossFunc(fakeResult, negs)
        discriminatorLoss = (realLoss + fakeLoss) / 2
        discriminatorLoss.backward()
        self.disciriminatorOptim.step()
        
        backwardTime = time.perf_counter() - startTime
        # loss = loss1 + tsLoss
        print("gLoss\t", format(generatorLoss.item(), ".7f"), "\tdLoss\t", format(discriminatorLoss.item(), ".7f"))
        return generatorLoss + discriminatorLoss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.generator.eval()
        self.staticFeatureModel.eval()
        self.tempFeatureModel.eval()
        for abnormalIdx in range(len(validsetLengths)):
            abnormalInputSet, abnormalLabelSet, abnormalLengths = self.generator.getInputTensor(
                validDataset, validsetLengths)
            zt = torch.tensor(numpy.random.normal(0, 1, (abnormalInputSet.shape[0], abnormalInputSet.shape[1], abnormalInputSet.shape[2])), dtype=torch.float32, device=torch.device('cuda'))
            tempFeat = self.tempFeatureModel(abnormalInputSet, abnormalLengths)
            staticFeat = self.staticFeatureModel(abnormalInputSet, abnormalLengths)
            staticFeat = staticFeat.repeat(1, abnormalInputSet.shape[1], 1)
            input = torch.cat((zt, staticFeat, tempFeat), 2)
            abnormalOutput = self.generator(input, abnormalLengths)    
            tl = abnormalOutput[abnormalIdx]
            t = abnormalLabelSet[abnormalIdx]
            ts = t - tl
            tlList = tl.reshape([-1]).tolist()
            tList = t.reshape([-1]).tolist()
            tsList = ts.reshape([-1]).tolist()
            # selftsList = self.ts[0].reshape([-1]).tolist()
            maxDiff = (torch.abs(abnormalLabelSet - abnormalOutput)).max().item()
            print("max diff\t", maxDiff)
            self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.taskName+ '-raetrainer-' + storeName + "-" + str(abnormalIdx))

    def save(self, filename=None):
        # torch.save(self.ts, self.raeTsPath)
        torch.save(self.generator.state_dict(), path.join(self.modelFolderPath, self.taskName +"-generator" + ".pt"))
        torch.save(self.discriminator.state_dict(), path.join(self.modelFolderPath, self.taskName +"-discriminator" + ".pt"))
        torch.save(self.staticFeatureModel.state_dict(), path.join(self.modelFolderPath, self.taskName +"-staticFeat" + ".pt"))
        torch.save(self.tempFeatureModel.state_dict(), path.join(self.modelFolderPath, self.taskName +"-tempFeat" + ".pt"))