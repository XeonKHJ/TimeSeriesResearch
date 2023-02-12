from Trainers.ITrainer import ITrainer
import torch
import time
import os.path as path

from globalConfig import globalConfig

class GeneratedRAETrainer(ITrainer):
    def __init__(self, mlModel, errorModel, logger, modelName, tsLambda=1e-6):
        self.aeModel = mlModel
        self.errorModel = errorModel
        self.lossFunc = torch.nn.MSELoss()
        self.tsLambda = tsLambda
        self.mlOptimzer = torch.optim.Adam(self.aeModel.parameters(), lr=1e-3)
        self.errorOptimzer = torch.optim.Adam(self.errorModel.parameters(), lr=1e-3)
        # self.raeTsPath = "SavedModels/raels.pt"
        self.epoch = 0
        self.logger = logger
        self.modelName = modelName

    def train(self, trainSet, trainSetLength, labelSet):
        self.aeModel.train()
        self.errorModel.train()
    
        startTime = time.perf_counter()
        ts = self.errorModel(trainSet, trainSetLength)
        tl = self.aeModel(trainSet, trainSetLength)
        t = tl+ts
        
        ts = self.errorModel(trainSet, trainSetLength)
        tl = trainSet - ts
        x = self.aeModel(tl, trainSetLength)
        fowardTime = time.perf_counter() - startTime
        loss1 = self.lossFunc(tl, x) + self.tsLambda * (1/torch.norm(ts, p=1))
        loss1.backward()
        self.mlOptimzer.step()
        self.errorOptimzer.step()
        self.mlOptimzer.zero_grad()
        self.errorOptimzer.step()
        
        tl = self.aeModel(tl, trainSetLength).detach()
        ts2 = self.errorModel(tl, trainSetLength)
        loss2 = self.tsLambda * torch.norm(ts2, p=1)
        loss2.backward()
        self.errorOptimzer.step()
        self.errorOptimzer.zero_grad()
        self.mlOptimzer.zero_grad()       

        backwardTime = time.perf_counter() - startTime
        loss = loss1 + loss2
        print("loss\t", format(loss.item(), ".7f"), "\t", format(loss1.item(), ".7f"), "\t", format(loss2.item() / self.tsLambda, ".7f"), "\tfoward\t", format(fowardTime, ".3f"), "\tbackward\t", format(backwardTime, ".3f"))
        return loss

    def evalResult(self, validDataset, validsetLengths, labels):
        self.aeModel.eval()
        reconstructData = self.aeModel(validDataset, validsetLengths)
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
        self.aeModel.eval()
        lengths = lengths.int()
        for validIdx in range(len(lengths)):
            validOutput = self.reconstruct(self.aeModel, dataset, lengths)
            for featIdx in range(dataset.shape[2]):
                tl = validOutput[validIdx,:,featIdx]
                t = dataset[validIdx,:,featIdx]
                ts = t - tl
                tlList = tl.reshape([-1])[0:lengths[validIdx]].tolist()
                tList = t.reshape([-1])[0:lengths[validIdx]].tolist()
                tsList = ts.reshape([-1])[0:lengths[validIdx]].abs().tolist()
                self.logger.logResults([tList, tsList, tlList], ["t", "ts", "tl"], self.modelName + '-' + storeNames[validIdx] + '-feat' + str(featIdx))

    def save(self):
        # torch.save(self.ts, self.raeTsPath)
        torch.save(self.aeModel.state_dict(), path.join(globalConfig.getModelPath(), self.modelName + "-ae.pt"))
        torch.save(self.errorModel.state_dict(), path.join(globalConfig.getModelPath(), self.modelName + "-error.pt"))

    def load(self):
        self.aeModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), self.modelName+"-ae.pt")))
        self.errorModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), self.modelName + "-error.pt")))

    def reconstruct(self, mlModel, validDataset, validsetLength):
        return mlModel(validDataset, validsetLength)