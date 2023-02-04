import torch
import torch.nn
import os.path as path
from DatasetReader.HSSReader import HSSReader
from DatasetReader.SegmentNABDataReader import SegmentNABDataReader
from DatasetReader.SingleNABDataReader import SingleNABDataReader
from TaskConfig.CorrectTaskConfig import CorrectTaskConfig
from TaskConfig.GruAEConfig import GruAEConfig
from TaskConfig.OffsetGruAEConfig import OffsetGruAEConfig
from TaskConfig.RAECorrectorTaskConfig import RAECorrectorTaskConfig
from TaskConfig.RAECorrectorWithTrendTaskConfig import RAECorrectorWithTrendTaskConfig


from TaskConfig.RAETaskConfig import RAETaskConfig
from DatasetReader.NABReader import NABReader
import ArgParser
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig
from TaskConfig.StaticAeConfig import StaticAeConfig
from TaskConfig.RandomRAETaskConfig import RandomRAETaskConfig
from TaskConfig.TimeGanConfig import TimeGanConfig
from Trainers.CorrectorTrainer import CorrectorTrainer

# normalDataReader = NABReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly")
# normalDataReader = SegmentNABDataReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv")
normalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv")
# normalDataReader = HSSReader("../datasets/preprocessed/HSS")
# abnormalDataReader = HSSReader("../datasets/preprocessed/HSS", isNormal=False)
abnormalDataReader = NABReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly")
# abnormalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv")
# skabDataReader = SKABDatasetReader("C:\\Users\\redal\\source\\repos\\SKAB\\data\\valve1")
modelFolderPath = "SavedModels"

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is avaliable.")
    else:
        print("CUDA is unavaliable")

    # read arg
    args = ArgParser.getArgs()
    isLoggerEnable = not (args.disablePlot)

    # load data
    normalDataset, normalDatasetLengths, normalLabels = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths, abnormalLabels = abnormalDataReader.read()

    # config = TimeGanConfig(modelFolderPath, isLoggerEnable)
    config = RandomRAETaskConfig(modelFolderPath, isLoggerEnable)
    # config = RAETaskConfig(modelFolderPath, isLoggerEnable)
    # config = RAECorrectorTaskConfig(modelFolderPath)
    # config = OneDAutoencoderConfig(modelFolderPath, isLoggerEnable)
    # config = StaticAeConfig(modelFolderPath, isLoggerEnable)
    # config = RAECorrectorWithTrendTaskConfig(modelFolderPath, isLoggerEnable)
    # config = OffsetGruAEConfig(modelFolderPath, isLoggerEnable, len(normalDataset[0][0]), len(normalDataset[0][0]))
    # config = GruAEConfig(modelFolderPath, isLoggerEnable)

    # load config
    mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName = config.getConfig()


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
    while epoch <= 10000:
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
            # trainer.evalResult(normalDataset, normalDatasetLengths, 'normalset')
            trainer.evalResult(abnormalDataset, abnormalDatasetLengths, 'abnormalset')
            trainer.save()
        epoch += 1

