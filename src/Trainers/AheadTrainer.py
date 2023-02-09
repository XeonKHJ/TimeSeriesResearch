from Trainers.ITrainer import ITrainer
import time
import torch
import os.path as path

class AheadTrainer(ITrainer):
    def __init__(self, model, taskName, logger, learningRate=1e-3, fileList=[]):
        self.mlModel = model
        self.lossFunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=learningRate)
        self.taskName = taskName
        self.logger = logger
        self.fileList = fileList
        self.modelFolderPath = "SavedModels"
        self.windowSize = 100
        self.splitData = None

    def train(self, trainSet, trainSetLength, labelSet):
        self.mlModel.train()
        startTime = time.perf_counter()
        windowSize = self.windowSize
        
        if self.splitData == None:
            self.splitedData = list()
            # split
            for idx in range(trainSet.shape[0]):
                splitdata = list()
                for timeIdx in range(trainSet.shape[1] - self.windowSize + 1):
                    splitdata.append(trainSet[idx, timeIdx:timeIdx+self.windowSize, :])
                self.splitedData.append(torch.cat(splitdata).reshape((-1, self.windowSize, trainSet.shape[2])))
            # self.splitedData = torch.cat(self.splitedData)

        for setIdx in range(trainSet.shape[0]):
            lengths = torch.tensor(windowSize).repeat(self.splitedData[setIdx].shape[0]).tolist()
            output = self.mlModel(self.splitedData[setIdx], lengths)
            outputedTime = time.perf_counter()
            output = output[0:output.shape[0]-self.windowSize]
            output = torch.cat((self.splitedData[setIdx][0:self.windowSize], output))
            loss = self.lossFunc(output, self.splitedData[setIdx])
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
        preIdx = -100
        for idx in range(0, validDataset.shape[1] - self.windowSize, self.windowSize):
            if idx+2*self.windowSize > reconstructSeqs.shape[1]:
                break
            lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).tolist()
            reconstructSeqs[:,idx+self.windowSize:idx+2*self.windowSize,:] = mlModel(validDataset[:,idx:idx+self.windowSize,:], lengths)
            preIdx = idx
            
        lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).tolist()
        reconstructSeqs[:,reconstructSeqs.shape[1]-self.windowSize:reconstructSeqs.shape[1],:] = mlModel(validDataset[:,validDataset.shape[1]-2*self.windowSize:validDataset.shape[1]-self.windowSize:,:], lengths)
        reconstructSeqs[:,0:self.windowSize, :] = validDataset[:, 0:self.windowSize, :]
        return reconstructSeqs