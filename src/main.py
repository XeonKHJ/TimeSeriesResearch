import torch
import torch.nn
import os.path as path
from Network.CUDASeasonalityLstmAutoencoder import CUDASeasonalityLstmAutoencoder
from DataNormalizer.DataNormalizer import DataNormalizer
from DataNormalizer.NoDataNormalizer import NoDataNormalizer
from DatasetReader.SmallNABReader import SmallNABReader
from Logger.PlotLogger import PlotLogger
from Logger.SimpleLogger import SimpleLogger
from Network.LstmAutoencoderWithCorrector import LstmAutoencoderWithCorrector
from Network.LstmAutoencoder import LstmAutoencoder
from Network.OffsetTwowayRNN import OffsetTwowayRNN
from Network.TraditionLstm import TraditionLstm
from DatasetReader.NABReader import NABReader
from DatasetReader.SKABDatasetReader import SKABDatasetReader
from Network.TwowayRNN import TwowayRNN
from Network.OffsetBiLstmAutoencoder import OffsetBiLstmAutoencoder

from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from Network.SeasonalityModel import SeasonalityModel
from Network.SeasonalityLstmAutoencoder import SeasonalityLstmAutoencoder
from Trainers.RAETrainer import RAETrainer
from Trainers.Trainer import Trainer

normalDataReader = NABReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly")
abnormalDataReader = NABReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly")
# skabDataReader = SKABDatasetReader("C:\\Users\\redal\\source\\repos\\SKAB\\data\\valve1")
modelFolderPath = "SavedModels"

def getConfig():
    feature_size = 1
    output_size = 1

    mlModel = CUDASeasonalityLstmAutoencoder(feature_size,4,output_size,2).cuda()
    try:
        mlModel.load_state_dict(torch.load(path.join(modelFolderPath, mlModel.getName() + ".pt")))
    except:
        pass
    
    trainer = Trainer(mlModel)
    datasetSeperator = NoSepDataSeperator()
    # logger = PlotLogger()
    logger = PlotLogger()
    dataNormalizer = DataNormalizer()
    return mlModel, datasetSeperator, trainer, logger, dataNormalizer

if __name__ == '__main__':
    # dataset, datasetLengths = datasetReader.read()
    normalDataset, normalDatasetLengths = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths = abnormalDataReader.read()
    # skabDataReader, skabDataLengths = skabDataReader.read()

    isLoggerEnable = True
    

    mlModel, datasetSeperator, trainer, logger, dataNormalizer = getConfig()
    
    dataNormalizer.addDatasetToRef(normalDataset)
    dataNormalizer.addDatasetToRef(abnormalDataset)
    normalDataset = dataNormalizer.normalizeDataset(normalDataset)
    abnormalDataset = dataNormalizer.normalizeDataset(abnormalDataset)
    trainDataset = datasetSeperator.getTrainningSet(normalDataset)
    trainsetLengths = datasetSeperator.getTrainningSet(normalDatasetLengths)

    validDataset = datasetSeperator.getValidationSet(normalDataset)
    validsetLengths = datasetSeperator.getValidationSet(normalDatasetLengths)

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
        print("epoch\t",epoch,"\tloss\t", loss.item())
        if epoch % 100 == 0:
            if isLoggerEnable:
                mlModel.eval()
                validInput, validOutput, validLengthes = mlModel.getInputTensor(validDataset, validsetLengths)
                abInput, abOutput, abLengths = mlModel.getInputTensor(abnormalDataset, abnormalDatasetLengths)
                anaOutput = mlModel(validInput, validLengthes)
                anaAbnormalOutput = mlModel(abInput, abLengths)
                print("result\t", torch.mean(anaOutput).item(), "\t", torch.mean(anaAbnormalOutput).item(), "\t", loss.item())
                x = validOutput[1].reshape([-1]).tolist()
                px = anaOutput[1].reshape([-1]).tolist()

                abx = abOutput[3].reshape([-1]).tolist()
                abpx = anaAbnormalOutput[3].reshape([-1]).tolist()
                logger.logResult(abx, [])
                logger.logResult(x, px)
                logger.logResult(abx, abpx)
            torch.save(mlModel.state_dict(), path.join(modelFolderPath, mlModel.getName() + ".pt"))
            mlModel.train()
        epoch += 1