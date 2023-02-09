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
        loss = self.lossFunc(output, trainSet)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        backwardTime = time.perf_counter()
        print("\tloss\t", loss.item(), "\tforward\t", outputedTime-startTime, "\tlosstime\t", backwardTime-outputedTime)
        return loss

    def evalResult(self, validDataset, validsetLengths, storeName=None):
        self.mlModel.eval()
        for validIdx in range(len(validsetLengths)):
            validOutput = self.reconstruct(self.mlModel, validDataset, validsetLengths)
            if validIdx<len(self.fileList):
                ogFileName = "-"+ path.splitext(path.basename(self.fileList[validIdx]))[0]
            else:
                ogFileName = ""
            for featIdx in range(validDataset.shape[2]):
                tl = validOutput[validIdx,:,featIdx]
                t = validDataset[validIdx,:,featIdx]
                ts = t - tl
                tlList = tl.reshape([-1])[0:validsetLengths[validIdx]].tolist()
                tList = t.reshape([-1])[0:validsetLengths[validIdx]].tolist()
                tsList = ts.reshape([-1])[0:validsetLengths[validIdx]].abs().tolist()
                # maxDiff = (torch.abs(abnormalLabelSet - validOutput)).max().item()
                # print("max diff\t", maxDiff)
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.taskName + '-' + storeName + "-idx" + str(validIdx) + '-feat' + str(featIdx) +ogFileName )

    def save(self, filename=None):
        if filename == None:
            filename = self.taskName + ".pt"
        else:
            filename = self.taskName + '-' + filename + ".pt"
        torch.save(self.mlModel.state_dict(), path.join(self.modelFolderPath, filename))

    def reconstruct(self, mlModel, validDataset, validsetLength):
        reconstructSeqs = torch.zeros(validDataset.shape, device=torch.device('cuda'))
        for idx in range(0, validDataset.shape[1]-100+1, 50):
            curInput = validDataset[:,idx:idx+100,:]
            lengths = torch.tensor(curInput.shape[1]).repeat(curInput.shape[0])
            output = mlModel(validDataset[:,idx:idx+100,:], lengths)
            reconstructSeqs[:,idx:idx+100,:] = output
        return reconstructSeqs