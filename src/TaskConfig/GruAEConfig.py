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
    def __init__(self, modelFolderPath, isLogEnable):
        self.modelFolderPath = modelFolderPath
        self.isLogEnable = isLogEnable

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        logger = PlotLogger(isPlotEnable=self.isLogEnable)
        mlModel = GruAutoencoder(feature_size,4,output_size,2)
        taskName = 'GruAEConfig'
        try:
            mlModel.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + ".pt")))
        except:
            pass
        if torch.cuda.is_available():
            mlModel.cuda()
        # trainer = RAETrainer(mlModel, logger, taskName, 10)
        trainer = Trainer(mlModel, taskName, logger, 1e-3)
        datasetSeperator = NoSepDataSeperator()
        # logger = PlotLogger()
        
        dataNormalizer = DataNormalizer()
        
        return mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName
