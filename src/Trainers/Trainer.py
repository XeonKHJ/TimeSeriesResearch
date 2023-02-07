from Trainers.ITrainer import ITrainer
import time
import torch
import os.path as path

class Trainer(ITrainer):
    def __init__(self, model, taskName, logger, learningRate=1e-3, fileList=[]):
        self.mlModel = model
        self.lossFunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=learningRate)
        self.taskName = taskName
        self.logger = logger
        self.fileList = fileList
        self.modelFolderPath = "SavedModels"

    def train(self, trainSet, trainSetLength, labelSet):
        self.mlModel.train()
        startTime = time.perf_counter()
        output = self.mlModel(trainSet, trainSetLength)
        outputedTime = time.perf_counter()
        loss = self.lossFunc(output, labelSet)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        backwardTime = time.perf_counter()
        print("\tloss\t", loss.item(), "\tforward\t", outputedTime-startTime, "\tlosstime\t", backwardTime-outputedTime)
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.mlModel.eval()
        for abnormalIdx in range(len(validsetLengths)):
            abnormalInputSet, abnormalLabelSet, abnormalLengths = self.mlModel.getInputTensor(
                validDataset, validsetLengths)
            abnormalOutput = self.mlModel(abnormalInputSet, abnormalLengths) 
            if abnormalIdx<len(self.fileList):
                ogFileName = "-"+ path.splitext(path.basename(self.fileList[abnormalIdx]))[0]
            else:
                ogFileName = ""
            for featIdx in range(validDataset.shape[2]):
                tl = abnormalOutput[abnormalIdx,:,featIdx]
                t = abnormalLabelSet[abnormalIdx,:,featIdx]
                ts = t - tl
                tlList = tl.reshape([-1])[0:validsetLengths[abnormalIdx]].tolist()
                tList = t.reshape([-1])[0:validsetLengths[abnormalIdx]].tolist()
                tsList = ts.reshape([-1])[0:validsetLengths[abnormalIdx]].abs().tolist()
                maxDiff = (torch.abs(abnormalLabelSet - abnormalOutput)).max().item()
                # print("max diff\t", maxDiff)
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.taskName + '-' + storeName + "-idx" + str(abnormalIdx) + '-feat' + str(featIdx) +ogFileName )

    def save(self, filename=None):
        if filename == None:
            filename = self.taskName + ".pt"
        else:
            filename = self.taskName + '-' + filename + ".pt"
        torch.save(self.mlModel.state_dict(), path.join(self.modelFolderPath, filename))