from Logger.PlotLogger import PlotLogger
from Network.LstmAutoencoder import LstmAutoencoder
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.CorrectorTrainer import CorrectorTrainer
from Trainers.RAETrainer import RAETrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class CorrectTaskConfig(ITaskConfig):
    def __init__(self, modelFolderPath, detectModel):
        self.modelFolderPath = modelFolderPath
        self.detectModel = detectModel

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        logger = PlotLogger()
        mlModel = LstmAutoencoder(feature_size,4,output_size,2)
        taskName = 'CorrectTask'
        try:
            mlModel.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + ".pt")))
        except:
            pass
        if torch.cuda.is_available():
            mlModel.cuda()
        trainer = CorrectorTrainer(self.detectModel, mlModel ,logger)
        datasetSeperator = NoSepDataSeperator()
        dataNormalizer = DataNormalizer()
        
        return mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName
