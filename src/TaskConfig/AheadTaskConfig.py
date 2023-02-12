from Network.BiGruAutoencoder import BiGruAutoencoder
from Network.GruAutoencoder import GruAutoencoder
from Logger.PlotLogger import PlotLogger
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.AheadTrainer import AheadTrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class AheadTaskConfig(ITaskConfig):
    def __init__(self, logger, modelName, showTrainingInfo=True):
        self.logger = logger
        self.modelName = modelName
        self.showTrainningInfo = showTrainingInfo

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        mlModel = BiGruAutoencoder(feature_size,10,output_size,4)
        if torch.cuda.is_available():
            mlModel.cuda()
        trainer = AheadTrainer(mlModel, self.modelName, self.logger, 1e-3, self.showTrainningInfo)
        try:
            trainer.load()
        except:
            pass
        
        return trainer
