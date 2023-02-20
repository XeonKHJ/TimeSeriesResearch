from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor
from Trainers.ITrainer import ITrainer
import time
import torch
import os.path as path

from globalConfig import globalConfig

class AheadTrainer(ITrainer):
    def __init__(self, model, taskName, logger, learningRate=1e-3, showTrainningInfo=True):
        self.mlModel = model
        self.lossFunc = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.mlModel.parameters(), lr=learningRate)
        self.modelName = taskName
        self.logger = logger
        self.windowSize = 100
        self.step = 1
        self.splitData = None
        self.showTrainningInfo = showTrainningInfo

    def train(self, trainSet, lengths, labelSet):
        self.mlModel.train()
        startTime = time.perf_counter()
        preSet = trainSet[:,0:int(trainSet.shape[1]/2), :]
        latterSet = trainSet[:,int(trainSet.shape[1]/2):trainSet.shape[1], :]

        output = self.mlModel(preSet, lengths / 2)
        loss = self.lossFunc(output, latterSet)
        outputedTime = time.perf_counter()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        backwardTime = time.perf_counter()
        print("\tloss\t", loss.item(), "\tforward\t", outputedTime-startTime, "\tlosstime\t", backwardTime-outputedTime)

        return loss

    def evalResult(self, validDataset, validsetLengths, labels):
        self.mlModel.eval()
        reconstructData = self.reconstruct(self.mlModel, validDataset, validsetLengths)
        error = torch.abs(reconstructData-validDataset)
        evalWindowSize = 100
        for threadhold in [0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001, 0.0005, 0.0001]:
            truePositive = 0
            falsePostive = 0
            falseNegative = 0
            step = 20
            for evalIdx in range(0, len(validsetLengths)):
                evalOutput = error[evalIdx, 0:validsetLengths[evalIdx].int().item()]
                detectResult = evalOutput > threadhold
                for labelIdx in range(0, validsetLengths[evalIdx].int().item(), evalWindowSize):
                    realPosCount = 0
                    predPosCount = 0

                    labelSegement = labels[evalIdx, labelIdx:labelIdx+evalWindowSize]
                    realPosCount = torch.sum(labelSegement)

                    knownDataLabel = labels[:, evalIdx:evalIdx + evalWindowSize]
                    evalBeginIdx = evalIdx + step - evalWindowSize
                    evalEndIdx = evalIdx + evalWindowSize - step

                    for rangeIdx in range(evalBeginIdx, evalEndIdx, step):
                        if rangeIdx >= 0 and rangeIdx < evalEndIdx:
                            diff = detectResult[rangeIdx:rangeIdx+evalWindowSize]
                            predPosCount += torch.sum(diff).int().item()

                    # If a known anomalous window overlaps any predicted windows, a TP is recorded.
                    if realPosCount != 0 and predPosCount != 0:
                        truePositive += 1

                    # If a known anomalous window does not overlap any predicted windows, a FN is recorded.
                    elif realPosCount != 0 and predPosCount == 0:
                        falseNegative += 1

                for predIdx in range(0, evalOutput.shape[1], evalWindowSize):
                    realPosCount = 0
                    predPosCount = 0
                    diff = detectResult[rangeIdx:rangeIdx+evalWindowSize]
                    predPosCount = torch.sum(diff).int().item()
                    evalBeginIdx = evalIdx + step - evalWindowSize
                    evalEndIdx = evalIdx + evalWindowSize - step

                    for rangeIdx in range(evalBeginIdx, evalEndIdx, step):
                        if rangeIdx >= 0 and rangeIdx < evalEndIdx:
                            realPosCount += torch.sum(labels[evalIdx, rangeIdx:rangeIdx+evalWindowSize]).int().item()
                    
                    # If a predicted window does not overlap any labeled anomalous region, a FP is recorded.
                    if predPosCount != 0 and realPosCount == 0:
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
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.modelName + '-' + storeNames[validIdx] + '-feat' + str(featIdx))

    def save(self):
        filename = self.modelName + ".pt"
        torch.save(self.mlModel.state_dict(), path.join(globalConfig.getModelPath(), filename))

    def load(self):
        filename = self.modelName + ".pt"
        self.mlModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), filename))) 

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