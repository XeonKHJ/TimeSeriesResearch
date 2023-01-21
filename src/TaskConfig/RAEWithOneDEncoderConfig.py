from Network.GruOneDEncodedAutoencoder import GruOneDEncodedAutoencoder
from Logger.PlotLogger import PlotLogger
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.Trainer import Trainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class RAEWithOneDEncoderConfig(ITaskConfig):
    def __init__(self, modelFolderPath, isLogEnable):
        self.modelFolderPath = modelFolderPath
        self.isLogEnable = isLogEnable

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        logger = PlotLogger(isPlotEnable=self.isLogEnable)
        mlModel = GruOneDEncodedAutoencoder(feature_size,4,output_size,2)
        taskName = 'RAEWithOneDEncoderConfig'
        try:
            mlModel.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + ".pt")))
        except:
            pass
        if torch.cuda.is_available():
            mlModel.cuda()
        trainer = Trainer(mlModel, taskName + '-raemodel', logger)
        datasetSeperator = NoSepDataSeperator()
        # logger = PlotLogger()
        
        dataNormalizer = DataNormalizer()
        
        return mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName
