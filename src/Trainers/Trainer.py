from Trainers.ITrainer import ITrainer
import time
import torch
import os.path as path
from globalConfig import globalConfig

class Trainer(ITrainer):
    def __init__(self, model, logger, learningRate=1e-3, exprimentName=None, showInfo=True):
        self.mlModel = model
        if torch.cuda.is_available():
            self.mlModel.cuda()
        self.lossFunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=learningRate)
        self.logger = logger
        self.experimentName = exprimentName
        self.showInfo = showInfo

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
        if self.showInfo:
            print("\tloss\t", format(loss.item(), ".9f"), "\tforward\t", format(outputedTime-startTime, ".7f"), "\tlosstime\t", format(backwardTime-outputedTime, ".7f"))
        return loss

    def evalResult(self, validDataset, validsetLengths, labels):
        self.mlModel.eval()
        validOutput = self.mlModel(validDataset, validsetLengths)
        diff = torch.abs(validOutput - validDataset)
        for threadhold in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001, 0.0005, 0.0001]:
            compareTensor = (diff > threadhold).reshape(diff.shape[0], -1)

            truePositive = 0
            falsePostive = 0
            falseNegative = 0
            evalWindowSize= 50
            for evalIdx in range(0, validOutput.shape[0]):
                curData = (compareTensor[evalIdx].bool()).sum(0) >= 1
                curLabel = (labels[evalIdx].bool()).sum(0) >= 1

                if curLabel.sum() > 0:
                    if curData.sum() > 0:
                        truePositive += curLabel.sum()
                # trueNegative += ((~(curData.bool())).sum() * (~(curLabel.bool())).sum()).bool().sum().item()

                temp = curLabel.sum().bool().int() - curData.sum().bool().int()
                falseNegative += (temp == 1).sum().item()
                falsePostive += (temp == -1).sum().item()

            precision = truePositive
            recall = truePositive
            f1 = 0
            if truePositive != 0:
                precision = truePositive / (truePositive + falsePostive)
                recall = truePositive / (truePositive + falseNegative)
                f1 = 2*(recall * precision) / (recall + precision)
            print('\tth\t', threadhold, '\teval\t', '\tprecision\t', format(precision, '.3f'), '\trecall\t', format(recall, '.3f'), '\tf1\t', format(f1, '.3f'))    

    def recordResult(self, dataset, lengths, storeNames):
        self.mlModel.eval()
        lengths = lengths.int()
        for validIdx in range(len(lengths)):
            validOutput = self.reconstruct(self.mlModel, dataset, lengths)
            for featIdx in range(dataset.shape[2]):
                tl = validOutput[validIdx,:,featIdx]
                t = dataset[validIdx,:,featIdx]
                ts = t - tl
                tlList = tl.reshape([-1])[0:lengths[validIdx]].tolist()
                tList = t.reshape([-1])[0:lengths[validIdx]].tolist()
                tsList = ts.reshape([-1])[0:lengths[validIdx]].abs().tolist()
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.experimentName + '-' + storeNames[validIdx] + '-feat' + str(featIdx))

    def save(self):
        filename = self.experimentName + ".pt"
        torch.save(self.mlModel.state_dict(), path.join(globalConfig.getModelPath(), filename))

    def load(self):
        filename = self.experimentName + ".pt"
        self.mlModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), filename)))

    def reconstruct(self, mlModel, validDataset, validsetLength):
        reconstructSeqs = torch.zeros(validDataset.shape, device=torch.device('cuda'))
        for idx in range(0, validDataset.shape[1]-100+1, 50):
            curInput = validDataset[:,idx:idx+100,:]
            lengths = torch.tensor(curInput.shape[1]).repeat(curInput.shape[0])
            output = mlModel(validDataset[:,idx:idx+100,:], lengths)
            reconstructSeqs[:,idx:idx+100,:] = output
        return reconstructSeqs