import torch
import torch.nn
import os.path as path
from DatasetReader.SingleNABDataReader import SingleNABDataReader
from TaskConfig.CorrectTaskConfig import CorrectTaskConfig
from TaskConfig.RAECorrectorTaskConfig import RAECorrectorTaskConfig


from TaskConfig.RAETaskConfig import RAETaskConfig
from DatasetReader.NABReader import NABReader
import ArgParser
from TaskConfig.RAEWithOneDEncoderConfig import RAEWithOneDEncoderConfig
from Trainers.CorrectorTrainer import CorrectorTrainer

normalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv")
# normalDataReader = NABReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly")
# abnormalDataReader = NABReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly")
abnormalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv")
# skabDataReader = SKABDatasetReader("C:\\Users\\redal\\source\\repos\\SKAB\\data\\valve1")
modelFolderPath = "SavedModels"

if __name__ == '__main__':
    # read arg
    args = ArgParser.getArgs()
    isLoggerEnable = not (args.disablePlot)
    # config = RAETaskConfig(modelFolderPath, isLoggerEnable)
    # config = RAECorrectorTaskConfig(modelFolderPath)
    config = RAEWithOneDEncoderConfig(modelFolderPath, isLoggerEnable)

    # load config
    mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName = config.getConfig()

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
    trainer.setAbnormal(abnormalDataset, abnormalDatasetLengths)

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
            trainer.evalResult(normalDataset, normalDatasetLengths, 'normalset')
            trainer.evalResult(abnormalDataset, abnormalDatasetLengths, 'abnormalset')
            trainer.save()
        epoch += 1

