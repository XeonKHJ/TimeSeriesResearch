from Network.IterGruAutoencoder import IterGruAutoencoder
from Network.LstmOneDEncodedAutoencoder import LstmOneDEncodedAutoencoder
from Network.GruOneDEncodedAutoencoder import GruOneDEncodedAutoencoder
from Logger.PlotLogger import PlotLogger
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.RAETrainer import RAETrainer
from Trainers.Trainer import Trainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class OneDAutoencoderConfig(ITaskConfig):
    def __init__(self, modelFolderPath, logger, experimentName, showTrainingInfo, lr=1e-3):
        self.modelFolderPath = modelFolderPath
        self.logger = logger
        self.modelName = experimentName
        self.showTrainingInfo = showTrainingInfo
        self.lr = lr

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        mlModel = IterGruAutoencoder(feature_size,4,output_size,2)
        trainer = Trainer(mlModel, self.logger, self.lr, self.modelName, self.showTrainingInfo)
        try:
            trainer.load()
        except:
            pass
        return trainer