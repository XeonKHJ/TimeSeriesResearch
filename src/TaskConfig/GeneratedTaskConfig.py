from Network.BiGruAutoencoder import BiGruAutoencoder
from Logger.PlotLogger import PlotLogger
from Network.IterGruAutoencoder import IterGruAutoencoder
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.GeneratedRAETrainer import GeneratedRAETrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class GeneratedTaskConfig(ITaskConfig):
    def __init__(self, logger, modelName, showTrainningInfo):
        self.modelName = modelName
        self.showTrainningInfo = showTrainningInfo
        self.logger = logger

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        mlModel = IterGruAutoencoder(feature_size,4,output_size,2)
        errorModel = BiGruAutoencoder(feature_size,4,output_size,2)
        if torch.cuda.is_available():
            mlModel.cuda()
            errorModel.cuda()
        trainer = GeneratedRAETrainer(mlModel, errorModel, self.logger, self.modelName)
        try:
            trainer.load()
        except:
            pass        

        return trainer
