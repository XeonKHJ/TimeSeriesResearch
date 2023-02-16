from Network.IterGruAutoencoder import IterGruAutoencoder
from Network.BiGruAutoencoder import BiGruAutoencoder
from Network.GruAutoencoder import GruAutoencoder
from Logger.PlotLogger import PlotLogger
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.AheadWithErrorGeneratorTrainer import AheadWithErrorGeneratorTrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class AheadWithErrorTaskConfig(ITaskConfig):
    def __init__(self, logger, modelName, showTrainingInfo=True):
        self.logger = logger
        self.modelName = modelName
        self.showTrainningInfo = showTrainingInfo

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        forcastModel = BiGruAutoencoder(feature_size,10,output_size,2)
        backwardModel = BiGruAutoencoder(feature_size,10,output_size,2)
        errorModel = IterGruAutoencoder(feature_size,10,output_size,2)
        if torch.cuda.is_available():
            forcastModel.cuda()
            errorModel.cuda()
            backwardModel.cuda()
        trainer = AheadWithErrorGeneratorTrainer(forcastModel,backwardModel,errorModel, self.modelName, self.logger, 1e-3, self.showTrainningInfo)
        try:
            trainer.load()
        except:
            pass
        
        return trainer
