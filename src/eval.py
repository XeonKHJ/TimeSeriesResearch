import torch
import torch.nn
import os.path as path

from Network.LstmAutoencoder import LstmAutoencoder

from Trainers.RAETrainer import RAETrainer
from Trainers.Trainer import Trainer

from DataNormalizer.DataNormalizer import DataNormalizer

from DatasetReader.SingleNABDataReader import SingleNABDataReader

from Logger.PlotLogger import PlotLogger

from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import ArgParser

normalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly/art_daily_small_noise.csv")
abnormalDataReader = SingleNABDataReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv")
modelFolderPath = "SavedModels"

def getConfig():
    feature_size = 1
    output_size = 1
    logger = PlotLogger()
    mlModel = LstmAutoencoder(feature_size,4,output_size,2)
    try:
        mlModel.load_state_dict(torch.load(path.join(modelFolderPath, mlModel.getName() + ".pt")))
    except:
        pass
    if torch.cuda.is_available():
        mlModel.cuda()
    trainer = RAETrainer(mlModel, logger)
    datasetSeperator = NoSepDataSeperator()
    dataNormalizer = DataNormalizer()
    return mlModel, datasetSeperator, trainer, logger, dataNormalizer

if __name__ == '__main__':

    args = ArgParser.getArgs()
    isLoggerEnable = not (args.disablePlot)

    normalDataset, normalDatasetLengths = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths = abnormalDataReader.read()

    mlModel, datasetSeperator, trainer, logger, dataNormalizer = getConfig()
    
    dataNormalizer.addDatasetToRef(normalDataset)
    dataNormalizer.addDatasetToRef(abnormalDataset)
    normalDataset = dataNormalizer.normalizeDataset(normalDataset)
    abnormalDataset = dataNormalizer.normalizeDataset(abnormalDataset)

    validDataset = datasetSeperator.getValidationSet(abnormalDataset)
    validsetLengths = datasetSeperator.getValidationSet(abnormalDatasetLengths)

    mlModel.eval()
    validInput, validOutput, validLengthes = mlModel.getInputTensor(validDataset, validsetLengths)
    output = mlModel(validInput, validLengthes)
    ts  = validInput - output
    outputRes = output[0].reshape([-1]).tolist()
    tsRes = ts[0].reshape([-1]).tolist()
    tRes = validInput[0].reshape([-1]).tolist()
    logger.logResults(
        [tsRes,tRes,outputRes], ['Ts', 'T', 'Tl']
    )