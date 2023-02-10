import torch
import torch.nn
import os.path as path
import numpy
from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

from Dataset.RegularDataset import RegularDataset
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.GruAEConfig import GruAEConfig


from TaskConfig.RAETaskConfig import RAETaskConfig
from DatasetReader.NABReader import NABReader
import ArgParser
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig
from TaskConfig.StaticAeConfig import StaticAeConfig
from TaskConfig.RandomRAETaskConfig import RandomRAETaskConfig
from TaskConfig.TimeGanConfig import TimeGanConfig
from Trainers.CorrectorTrainer import CorrectorTrainer

from torch.utils.data import DataLoader

windowSize = 100

modelFolderPath = "SavedModels"

def reconstruct(mlModel, validDataset, validsetLength):
    reconstructSeqs = torch.zeros(validDataset.shape, device=torch.device('cuda'))
    step = 50
    for idx in range(0, validDataset.shape[1]-windowSize+1, step):
        curInput = validDataset[:,idx:idx+windowSize,:]
        lengths = torch.tensor(curInput.shape[1]).repeat(curInput.shape[0])
        output = mlModel(validDataset[:,idx:idx+windowSize,:], lengths)
        reconstructSeqs[:,idx:idx+windowSize,:] = output

    # final
    timeLength = validDataset.shape[1]
    curInput = validDataset[:,timeLength-windowSize:timeLength,:]
    lengths = torch.tensor(curInput.shape[1]).repeat(curInput.shape[0])
    reconstructSeqs[:,timeLength-windowSize:timeLength,:] = mlModel(validDataset[:,idx:idx+windowSize,:], lengths)

    return reconstructSeqs
def evalutaion(mlModel, validDataset, validDatsetLengths, labels, loss):
    # 验证
    mlModel.eval()
    # noise = torch.tensor(numpy.random.normal(0, 1, (abnormalDataset.shape[0], abnormalDataset.shape[1], abnormalDataset.shape[2])), dtype=torch.float32, device=torch.device('cuda'))
    validDataset = validDataset.cuda()
    # evalSet = torch.cat((abnormalDataset, noise), 2)
    reconstructOutput = reconstruct(mlModel, validDataset, validDatsetLengths)

    for threadHole in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001, 0.0005, 0.0001]:
        compareTensor = torch.abs(reconstructOutput - validDataset)
        compareTensor = (compareTensor > threadHole)

        truePositive = 0
        falsePostive = 0
        falseNegative = 0
        trueNegative = 0
        evalWindowSize= 50
        for evalIdx in range(0, reconstructOutput.shape[1], evalWindowSize):
            curData = (compareTensor[:, evalIdx:evalIdx+evalWindowSize, :].bool()).sum(1) >= 1
            curLabel = (~labels[:, evalIdx:evalIdx+evalWindowSize, :].bool()).sum(1) >= 1

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
        print('loss\t', format(loss.item(), ".7f"), '\tth\t', threadHole, '\teval\t', '\tprecision\t', format(precision, '.3f'), '\trecall\t', format(recall, '.3f'), '\tf1\t', format(f1, '.3f'))    

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is avaliable.")
    else:
        print("CUDA is unavaliable")

    # read arg
    args = ArgParser.getArgs()
    isLoggerEnable = not (args.disablePlot)

    # setup dataset
    dataReader = NABFoldersReader("../../NAB/", "artificial")
    displayDataReader = NABFoldersReader("../../NAB/", "artificial")

    # load data
    fullDataTensor, fullDataLenghts, fullDataLabels, fileList = dataReader.read()

    # load config
    config = GruAEConfig(modelFolderPath, isLoggerEnable, fileList=fileList)
    mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName = config.getConfig()

    dataNormalizer.addDatasetToRef(fullDataTensor)
    fullDataTensor = dataNormalizer.normalizeDataset(fullDataTensor)
    fullDataTensor = torch.cat((fullDataTensor, fullDataLabels), 2)

    displayDataTensor, displayDataLenghts, displayDataLabels, displayFileList = dataReader.read()
    for i in range(displayDataTensor.shape[0]):
        curList = displayDataTensor[i].reshape(-1).tolist()
        logger.logResults([curList], ['real'], path.splitext(path.basename(fileList[i]))[0], "OgPics")

    # data preprocess
    processers = [
        SlidingWindowStepDataProcessor(windowSize=100, step=20),
        ShuffleDataProcessor()
    ]
    dataTensor = fullDataTensor
    dataLengths = fullDataLenghts
    for processor in processers:
        dataTensor, dataLengths = processor.process(dataTensor, dataLengths)
    trainDataTensor, trainsetLengths = datasetSeperator.getTrainningSet(dataTensor, dataLengths)
    validDataTensor, validsetLengths = datasetSeperator.getValidationSet(dataTensor, dataLengths)

    trainDataset = RegularDataset(trainDataTensor, trainsetLengths)
    validDataset = RegularDataset(validDataTensor, validsetLengths)

    # start trainning
    epoch = 0
    keepTrainning = True

    trainDataLoader = DataLoader(trainDataset, batch_size=1000, shuffle=True)
    testDataLoader = DataLoader(validDataset, shuffle=False, batch_size=validDataTensor.shape[0])

    while keepTrainning:
        for trainData, trainLabels in trainDataLoader:
            lengths = trainLabels[:, trainLabels.shape[1]-1]
            labels = trainLabels[:, 0:trainLabels.shape[1]-1]
            loss = trainer.train(trainData, lengths, labels)
        if epoch % 100 == 0:
            trainer.save()
            for testData, testLabels in testDataLoader:
                lengths = testLabels[:, testLabels.shape[1]-1]
                labels = testLabels[:, 0:testLabels.shape[1]-1]            
                trainer.evalResult(testData, lengths, labels)
        epoch += 1