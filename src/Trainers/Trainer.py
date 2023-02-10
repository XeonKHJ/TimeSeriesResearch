from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor
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
        evalWindowSize= 100
        dataProcessor = SlidingWindowStepDataProcessor(evalWindowSize, 1)
        
        toEvalSplitedDataTensorList = list()
        for idx in range(len(validsetLengths)):
            toEvalDataSplitedTensor, toEvalDataLength = dataProcessor.process(validDataset[idx:idx+1], [validsetLengths[idx]])
            toEvalOutputTensor = self.mlModel(toEvalDataSplitedTensor, toEvalDataLength)
            diff = torch.abs(toEvalOutputTensor - toEvalDataSplitedTensor)
            toEvalSplitedDataTensorList.append({"data":toEvalDataSplitedTensor, "length":toEvalDataLength, "output": toEvalOutputTensor})

        for threadhold in [0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001, 0.0005, 0.0001]:
            truePositive = 0
            falsePostive = 0
            falseNegative = 0
            step = 20
            for evalIdx in range(0, len(validsetLengths)):
                evalOutput = toEvalSplitedDataTensorList[evalIdx]["output"]
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
                            diff = detectResult[rangeIdx]
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
                    diff = detectResult[rangeIdx]
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