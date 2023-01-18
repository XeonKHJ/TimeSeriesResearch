import torch
import torch.nn
import os.path as path
from DatasetReader.SingleNABDataReader import SingleNABDataReader
from TaskConfig.CorrectTaskConfig import CorrectTaskConfig


from TaskConfig.RAETaskConfig import RAETaskConfig
from DatasetReader.NABReader import NABReader
import ArgParser
from Trainers.CorrectorTrainer import CorrectorTrainer

normalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv")
abnormalDataReader = NABReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly")
# skabDataReader = SKABDatasetReader("C:\\Users\\redal\\source\\repos\\SKAB\\data\\valve1")
modelFolderPath = "SavedModels"

config = RAETaskConfig(modelFolderPath)
correctTaskConfig = None
def getConfig():
    mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName = config.getConfig()
    return mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName

if __name__ == '__main__':
    # read arg
    args = ArgParser.getArgs()
    isLoggerEnable = not (args.disablePlot)

    # load config
    mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName = getConfig()

    # load data
    normalDataset, normalDatasetLengths = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths = abnormalDataReader.read()
    dataNormalizer.addDatasetToRef(normalDataset)
    dataNormalizer.addDatasetToRef(abnormalDataset)
    normalDataset = dataNormalizer.normalizeDataset(normalDataset)
    abnormalDataset = dataNormalizer.normalizeDataset(abnormalDataset)
    trainDataset = datasetSeperator.getTrainningSet(normalDataset)
    trainsetLengths = datasetSeperator.getTrainningSet(normalDatasetLengths)
    validDataset = datasetSeperator.getValidationSet(normalDataset)
    validsetLengths = datasetSeperator.getValidationSet(normalDatasetLengths)

    # start trainning
    toTrainDataset, labelDataset, labelDatasetLengths = mlModel.getInputTensor(trainDataset, trainsetLengths)
    mlModel.train()
    batchSize = toTrainDataset.shape[0]
    currentIdx = 0
    datasetSize = toTrainDataset.shape[0]
    epoch = 0
    while True:
        if datasetSize - batchSize == 0:
            startIdx = 0
        else:
            startIdx = currentIdx % (datasetSize - batchSize)
        endIdx = startIdx + batchSize
        currentIdx += batchSize 
        trainSet = toTrainDataset[startIdx:endIdx]
        labelSet = labelDataset[startIdx:endIdx]
        labelSetLengths = labelDatasetLengths[startIdx:endIdx]
        loss = trainer.train(trainSet, labelSetLengths, labelSet)
        if epoch % 100 == 0:
            if isLoggerEnable:
                trainer.evalResult(normalDataset, normalDatasetLengths, 'normalset')
                trainer.evalResult(abnormalDataset, abnormalDatasetLengths, 'abnormalLength')
            trainer.save()
        epoch += 1

