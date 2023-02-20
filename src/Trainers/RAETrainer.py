from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor
from Trainers.ITrainer import ITrainer
import torch
import time
import os.path as path

from globalConfig import globalConfig

class RAETrainer(ITrainer):
    def __init__(self, mlModel, logger, modelName, tsLambda=0.1):
        self.aeModel = mlModel
        self.lossFunc = torch.nn.MSELoss()
        self.ts = torch.zeros([0])
        self.tsLambda = tsLambda
        self.optimizer = torch.optim.Adam(self.aeModel.parameters(), lr=1e-3)
        self.step2Optimizer = None
        # self.raeTsPath = "SavedModels/raels.pt"
        self.epoch = 0
        self.logger = logger
        self.modelName = modelName

    def train(self, trainSet, trainSetLength, labelSet):
        self.aeModel.train()
        if self.ts.shape == torch.Size([0]):
            try:
                if torch.cuda.is_available():
                    print('CUDA is avaliable')
                    self.ts = torch.load(self.raeTsPath, map_location = torch.device('cuda'))
                else:
                    print('CUDA is not avaliable.')
                    self.ts = torch.load(self.raeTsPath)
            except:
                if torch.cuda.is_available():
                    self.ts = torch.zeros(trainSet.shape, requires_grad=True, device=torch.device('cuda'))
                else:
                    self.ts = torch.zeros(trainSet.shape, requires_grad=True)
            self.step2Optimizer = torch.optim.Adam([self.ts], lr=1e-3)
        startTime = time.perf_counter()
        tl = trainSet - self.ts
        x = self.aeModel(tl, trainSetLength)
        fowardTime = time.perf_counter() - startTime
        loss1 = self.lossFunc(tl, x)
        loss1.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        tl = self.aeModel(tl, trainSetLength)
        self.ts2 = trainSet - tl
        loss2 = self.tsLambda * torch.norm(self.ts2, p=1)
        loss2.backward()
        self.step2Optimizer.step()
        self.step2Optimizer.zero_grad()
        self.optimizer.zero_grad()
        

        backwardTime = time.perf_counter() - startTime
        loss = loss1 + loss2
        print("loss\t", format(loss.item(), ".7f"), "\t", format(loss1.item(), ".7f"), "\t", format(loss2.item() / self.tsLambda, ".7f"), "\tfoward\t", format(fowardTime, ".3f"), "\tbackward\t", format(backwardTime, ".3f"))
        return loss

    def evalResult(self, validDataset, validsetLengths, labels):
        self.aeModel.eval()
        reconstructData = self.aeModel(validDataset, validsetLengths)
        error = torch.abs(reconstructData-validDataset)

        evalWindowSize = 100

        slidingProcessor = SlidingWindowStepDataProcessor(evalWindowSize, 1)
        toEvalSplitedDataList = list()
        for idx in range(len(validsetLengths)):
            windowedData, toEvalDataLength = slidingProcessor.process(error[idx:idx+1], [validsetLengths[idx]])
            toEvalSplitedDataList.append({"data":windowedData, "length":toEvalDataLength})

        
        # windowedLabel, windowedLengths = slidingProcessor.process(labels, validsetLengths)
        for threadhold in [0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001, 0.0005, 0.0001]:
            truePositive = 0
            falsePostive = 0
            falseNegative = 0
            step = 20
            for evalIdx in range(0, len(validsetLengths)):
                evalOutput = toEvalSplitedDataList[evalIdx]["data"]
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

                for predIdx in range(0, detectResult.shape[0], evalWindowSize):
                    realPosCount = 0
                    predPosCount = 0
                    diff = detectResult[predIdx]
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
        torch.save(self.aeModel.state_dict(), path.join(globalConfig.getModelPath(), self.modelName + ".pt"))

    def load(self):
        self.aeModel.load_state_dict(torch.load(path.join(globalConfig.getModelPath(), self.modelName+".pt")))

    def reconstruct(self, mlModel, validDataset, validsetLength):
        return mlModel(validDataset, validsetLength)