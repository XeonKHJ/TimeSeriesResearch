from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor
from Trainers.ITrainer import ITrainer
import time
import torch
import os.path as path
from Utils import DynamicThreshold

from globalConfig import globalConfig

class AheadWithErrorGeneratorTrainer(ITrainer):
    def __init__(self, model, errorModel, taskName, logger, learningRate=1e-3, showTrainningInfo=True):
        self.forcastModel = model
        self.errorModel = errorModel
        self.lossFunc = torch.nn.MSELoss()
        self.forcastOptimizer = torch.optim.Adam(self.forcastModel.parameters(), lr=learningRate)
        self.errorOptimizer = torch.optim.Adam(self.errorModel.parameters(), lr=learningRate)
        self.modelName = taskName
        self.logger = logger
        self.windowSize = 100
        self.step = 1
        self.splitData = None
        self.showTrainningInfo = showTrainningInfo
        self.lambda1 = 1.5
        self.lambda2 = 1e-3

    def train(self, trainSet, lengths, labelSet):
        self.forcastModel.train()
        self.errorModel.train()
        startTime = time.perf_counter()
        preSet = trainSet[:,0:int(trainSet.shape[1]/2), :]
        latterSet = trainSet[:,int(trainSet.shape[1]/2):trainSet.shape[1], :]
        
        self.forcastOptimizer.zero_grad()
        self.errorOptimizer.zero_grad()
        error = self.errorModel(trainSet, lengths, int(trainSet.shape[1] / 2))
        output = self.forcastModel(preSet, lengths / 2)
        t = output + error
        loss2 = self.lossFunc(t, latterSet) + self.lambda2 * 1/torch.norm(error, p=1)
        loss2.backward()
        self.forcastOptimizer.step()
        self.errorOptimizer.step()

        self.forcastOptimizer.zero_grad()
        self.errorOptimizer.zero_grad()
        error = self.errorModel(trainSet, lengths, int(trainSet.shape[1] / 2))
        output = self.forcastModel(preSet, lengths / 2)
        tl = latterSet - error
        forcastLoss = self.lossFunc(output, tl)
        realLoss = self.lossFunc(output, latterSet)
        loss1 = forcastLoss + self.lambda1 * realLoss
        loss1.backward()
        self.forcastOptimizer.step()
        self.errorOptimizer.step()
        
        backwardTime = time.perf_counter()
        print("\tforcast loss\t", format(forcastLoss.item(), ".7f"), "\treal loss\t", format(realLoss.item(), ".7f"),"\tloss1\t", format(loss1.item(),".7f"),  "\tloss2\t", format(loss2.item(),".7f"), '\terror\t', torch.norm(error, p=1).item())

        return forcastLoss

    def evalResult(self, validDataset, validsetLengths, labels):
        self.forcastModel.eval()
        reconstructData = self.reconstruct(self.forcastModel, validDataset, validsetLengths)
        error = torch.abs(reconstructData-validDataset)
        evalWindowSize = 100
        threadholders = list()
        for threadhold in [0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001, 0.0005, 0.0001]:
            threadholders.append(DynamicThreshold(threadhold, evalWindowSize))
        for threadholder in threadholders:
            truePositive = 0
            falsePostive = 0
            falseNegative = 0
            step = 20
            thresholds = threadholder.getThreshold(validDataset)
            compareResult = threadholder.compare(thresholds, validDataset, reconstructData)
            for dataIdx in range(len(validsetLengths)):
                for windowIdx in range(0, validDataset.shape[1], evalWindowSize):
                    curLabel = (labels[dataIdx, windowIdx:windowIdx+evalWindowSize].sum() > 0).bool().item()
                    evalRes = (compareResult[dataIdx, windowIdx:windowIdx+evalWindowSize].sum()).bool().item()
                    if curLabel:
                        if evalRes:
                            truePositive += 1
                        else:
                            falseNegative += 1
                    else:
                        if evalRes:
                            falsePostive += 1

            
            precision = truePositive
            recall = truePositive
            f1 = 0
            if truePositive != 0:
                precision = truePositive / (truePositive + falsePostive)
                recall = truePositive / (truePositive + falseNegative)
                f1 = 2*(recall * precision) / (recall + precision)
            print('\tth\t', format(threadhold, '.5f'), '\tprecision\t', format(precision, '.5f'), '\trecall\t', format(recall, '.3f'), '\tf1\t', format(f1, '.5f')) 

    def recordResult(self, dataset, lengths, storeNames):
        self.forcastModel.eval()
        lengths = lengths.int()
        for validIdx in range(len(lengths)):
            validOutput = self.reconstruct(self.forcastModel, dataset, lengths)
            errorOutput = self.reconstructError(dataset, lengths)
            sumOutput = validOutput + errorOutput
            for featIdx in range(dataset.shape[2]):
                tl = validOutput[validIdx,:,featIdx]
                error = errorOutput[validIdx, :, featIdx]
                t = dataset[validIdx,:,featIdx]
                ts = t - tl
                tlList = tl.reshape([-1])[0:lengths[validIdx]].tolist()
                tList = t.reshape([-1])[0:lengths[validIdx]].tolist()
                tsList = ts.reshape([-1])[0:lengths[validIdx]].abs().tolist()
                errorList = error.reshape([-1])[0:lengths[validIdx]].tolist()
                sumList = sumOutput.reshape([-1])[0:lengths[validIdx]].tolist()
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.modelName + '-' + storeNames[validIdx] + '-forcast-' '-feat' + str(featIdx))
                self.logger.logResults([tList, errorList], ["t", "error"], self.modelName + '-' + storeNames[validIdx] + '-error-' '-feat' + str(featIdx))
                self.logger.logResults([tList, sumList], ["t", "sum"], self.modelName + '-' + storeNames[validIdx] + '-sum-' '-feat' + str(featIdx))

    def save(self):
        forcastModelName = self.modelName + "-forcast.pt"
        errorModelName = self.modelName + "-error.pt"
        torch.save(self.forcastModel.state_dict(), path.join(globalConfig.getModelPath(), forcastModelName))
        torch.save(self.errorModel.state_dict(), path.join(globalConfig.getModelPath(), errorModelName))

    def load(self):
        forcastModelName = self.modelName + "-forcast.pt"
        errorModelName = self.modelName + "-error.pt"
        self.forcastModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), forcastModelName))) 
        self.errorModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), errorModelName))) 

    def reconstruct(self, mlModel, validDataset, validsetLength):
        reconstructSeqs = torch.zeros(validDataset.shape, device=torch.device('cuda'))
        preIdx = -100
        for idx in range(0, validDataset.shape[1] - self.windowSize, self.windowSize):
            if idx+2*self.windowSize > reconstructSeqs.shape[1]:
                break
            lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
            reconstructSeqs[:,idx+self.windowSize:idx+2*self.windowSize,:] = mlModel(validDataset[:,idx:idx+self.windowSize,:], lengths)
            preIdx = idx
            
        lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
        reconstructSeqs[:,reconstructSeqs.shape[1]-self.windowSize:reconstructSeqs.shape[1],:] = mlModel(validDataset[:,validDataset.shape[1]-2*self.windowSize:validDataset.shape[1]-self.windowSize:,:], lengths)
        reconstructSeqs[:,0:self.windowSize, :] = validDataset[:, 0:self.windowSize, :]
        return reconstructSeqs

    def reconstructError(self, validDataset, validsetLength):
        reconstructSeqs = torch.zeros(validDataset.shape, device=torch.device('cuda'))
        preIdx = -100
        for idx in range(0, validDataset.shape[1] - 2 * self.windowSize, self.windowSize):
            if idx+2*self.windowSize > reconstructSeqs.shape[1]:
                break
            lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
            reconstructSeqs[:,idx+self.windowSize:idx+2*self.windowSize,:] = self.errorModel(validDataset[:,idx:idx+self.windowSize,:], lengths, self.windowSize)
            preIdx = idx
            
        lengths = torch.tensor(self.windowSize).repeat(validDataset.shape[0]).int()
        reconstructSeqs[:,0:self.windowSize, :] = 0
        return reconstructSeqs