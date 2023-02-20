from Logger.PlotLogger import PlotLogger
from Network.LstmAutoencoder import LstmAutoencoder
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.RAETrainer import RAETrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class RAETaskConfig(ITaskConfig):
    def __init__(self, logger, modelName, showTrainningInfo):
        self.modelName = modelName
        self.showTrainningInfo = showTrainningInfo
        self.logger = logger

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        mlModel = LstmAutoencoder(feature_size,4,output_size,2)
        if torch.cuda.is_available():
            mlModel.cuda()
        trainer = RAETrainer(mlModel, self.logger, self.modelName)
        try:
            trainer.load()
        except:
            pass        

        return trainer
