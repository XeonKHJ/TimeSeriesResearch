from Network.GruOneDEncodedAutoencoder import GruOneDEncodedAutoencoder
from Logger.PlotLogger import PlotLogger
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.RAETrainer import RAETrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class RAEWithOneDEncoderConfig(ITaskConfig):
    def __init__(self, modelFolderPath):
        self.modelFolderPath = modelFolderPath

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        logger = PlotLogger()
        mlModel = GruOneDEncodedAutoencoder(feature_size,4,output_size,4)
        taskName = 'RAEWithOneDEncoderConfig'
        try:
            mlModel.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + '-raemodel' + ".pt")))
        except:
            pass
        if torch.cuda.is_available():
            mlModel.cuda()
        trainer = RAETrainer(mlModel, logger, taskName + '-raemodel')
        datasetSeperator = NoSepDataSeperator()
        # logger = PlotLogger()
        
        dataNormalizer = DataNormalizer()
        
        return mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName
