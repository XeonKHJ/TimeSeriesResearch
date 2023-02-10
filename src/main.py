import torch
import torch.nn
import os.path as path
import numpy
from DataSegmentor.SlidingWindowStepDataSegement import SlidingWindowStepDataSegement
from DatasetReader.HSSReader import HSSReader
from DatasetReader.NABFoldersReader import NABFoldersReader
from DatasetReader.SegmentNABDataReader import SegmentNABDataReader
from DatasetReader.SegmentNABFolderDataReader import SegmentNABFolderDataReader
from DatasetReader.SegmentNABFoldersDataReader import SegmentNABFoldersDataReader
from DatasetReader.SingleNABDataReader import SingleNABDataReader
from TaskConfig.AheadTaskConfig import AheadTaskConfig
from TaskConfig.CorrectTaskConfig import CorrectTaskConfig
from TaskConfig.GruAEConfig import GruAEConfig
from TaskConfig.ItrGruAEConfig import ItrGruAEConfig
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

windowSize = 100
# normalDataReader = NABReader("../../NAB/data/artificialNoAnomaly")
normalDataReader = NABFoldersReader(["../../NAB/data/realAWSCloudwatch/"])
# normalDataReader = NABReader("../../NAB/data/realAWSCloudwatch/")
# normalDataReader = SegmentNABFolderDataReader("../../NAB/data/realAdExchange/", windowSize)
# normalDataReader = SegmentNABDataReader("../datasets/preprocessed//NAB/artificialWithAnomaly/artificialWithAnomaly/art_daily_flatmiddle.csv")
# normalDataReader = SegmentNABDataReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv")
# normalDataReader = SegmentNABFolderDataReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly")
# normalDataReader = SegmentNABFoldersDataReader(["../../NAB/data/artificialNoAnomaly","../../NAB/data/artificialWithAnomaly"], windowSize )
# normalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv")
# normalDataReader = SingleNABDataReader("../datasets/preprocessed//NAB/artificialWithAnomaly/artificialWithAnomaly/art_daily_flatmiddle.csv")
# normalDataReader = HSSReader("../datasets/preprocessed/HSS")
# abnormalDataReader = HSSReader("../datasets/preprocessed/HSS", isNormal=False)
reconstructDataReader = NABReader("../../NAB/data/realAWSCloudwatch/")
# abnormalDataReader = NABReader("../datasets/preprocessed//NAB/artificialWithAnomaly/artificialWithAnomaly")
# abnormalDataReader = NABFoldersReader(["../../NAB/data/artificialNoAnomaly","../../NAB/data/artificialWithAnomaly"])
# abnormalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv")
# skabDataReader = SKABDatasetReader("C:\\Users\\redal\\source\\repos\\SKAB\\data\\valve1")
modelFolderPath = "SavedModels"

def shuffle(dataset):
    shuffleIdx = torch.randperm(dataset.shape[0])
    dataset = dataset[shuffleIdx, :, :]
    return dataset

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

    # load data
    normalDataset, normalDatasetLengths, normalLabels, labels, _ = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths, abnormalLabels, abnormallabels, fileList = reconstructDataReader.read()

    segementor = SlidingWindowStepDataSegement(100, 20)
    segementedDataset = segementor.segement(normalDataset, normalDatasetLengths)
    shuffledDataset = shuffle(segementedDataset)

    # config = TimeGanConfig(modelFolderPath, isLoggerEnable)
    # config = RandomRAETaskConfig(modelFolderPath, isLoggerEnable)
    # config = RAETaskConfig(modelFolderPath, isLoggerEnable)
    # config = RAECorrectorTaskConfig(modelFolderPath)
    # config = OneDAutoencoderConfig(modelFolderPath, isLoggerEnable)
    # config = StaticAeConfig(modelFolderPath, isLoggerEnable)
    # config = RAECorrectorWithTrendTaskConfig(modelFolderPath, isLoggerEnable)
    # config = OffsetGruAEConfig(modelFolderPath, isLoggerEnable, len(normalDataset[0][0]), len(normalDataset[0][0]), fileList)
    config = GruAEConfig(modelFolderPath, isLoggerEnable, fileList=fileList)
    # config = ItrGruAEConfig(modelFolderPath, isLoggerEnable, fileList=fileList)
    # config = AheadTaskConfig(modelFolderPath, isLoggerEnable, fileList=fileList)

    # load config
    mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName = config.getConfig()

    dataNormalizer.addDatasetToRef(normalDataset)
    # dataNormalizer.addDatasetToRef(abnormalDataset)
    normalDataset = dataNormalizer.normalizeDataset(normalDataset)
    abnormalDataset = dataNormalizer.normalizeDataset(abnormalDataset)
    trainDataset, trainsetLengths = datasetSeperator.getTrainningSet(normalDataset, normalDatasetLengths)
    validDataset, validsetLengths = datasetSeperator.getValidationSet(normalDataset, normalDatasetLengths)
    trainer.setAbnormal(abnormalDataset, abnormalDatasetLengths)

    # save pic
    for i in range(abnormalDataset.shape[0]):
        curList = abnormalDataset[i].reshape(-1).tolist()
        logger.logResults([curList], ['real'], path.splitext(path.basename(fileList[i]))[0], "OgPics")

    # start trainning
    # toTrainDataset, labelDataset, labelDatasetLengths = mlModel.getInputTensor(trainDataset, trainsetLengths)
    mlModel.train()
    batchSize = trainDataset.shape[0]
    currentIdx = 0
    datasetSize = trainDataset.shape[0]
    epoch = 0
    keepTrainning = True
    while keepTrainning:
        if datasetSize - batchSize == 0:
            startIdx = 0
        else:
            startIdx = currentIdx % (datasetSize - batchSize)
        endIdx = startIdx + batchSize
        currentIdx += batchSize 
        trainSet = trainDataset[startIdx:endIdx]
        trainningLengths = trainsetLengths[startIdx:endIdx]
        labelSet = labels[startIdx:endIdx]
        loss = trainer.train(trainSet, trainningLengths, labelSet)
        if epoch % 1000 == 0:
            # trainer.evalResult(normalDataset, normalDatasetLengths, 'normalset')
            trainer.evalResult(abnormalDataset, abnormalDatasetLengths, 'abnormalset')
            trainer.save()
            evalutaion(mlModel, abnormalDataset, abnormalDatasetLengths, abnormallabels, loss)
        epoch += 1
