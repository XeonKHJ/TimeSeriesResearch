from Network.GruAutoencoder import GruAutoencoder
from Logger.PlotLogger import PlotLogger
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.RAETrainer import RAETrainer
from Trainers.Trainer import Trainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class GruAEConfig(ITaskConfig):
    def __init__(self, modelFolderPath, logger, modelName, inputFeatureCount=1, outputFeatureCount=1, fileList=[], showTrainingInfo=True):
        self.modelFolderPath = modelFolderPath
        self.logger = logger
        self.inputFeatureSize = inputFeatureCount
        self.outputFeatureSize = outputFeatureCount
        self.fileList = fileList
        self.modelName = modelName
        self.showTrainingInfo = showTrainingInfo

    def getConfig(self, isCuda = False):
        feature_size = self.inputFeatureSize
        output_size = self.outputFeatureSize
        
        mlModel = GruAutoencoder(feature_size,4,output_size,2)
        trainer = Trainer(mlModel, self.logger, 1e-3, self.modelName, self.showTrainingInfo)
        try:
            trainer.load()
        except:
            pass
        
        return trainer
