from Logger.PlotLogger import PlotLogger
from Network.LstmAutoencoder import LstmAutoencoder
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.RAETrainer import RAETrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class RAETaskConfig(ITaskConfig):
    def __init__(self, modelFolderPath, isPlotEnable=False):
        self.modelFolderPath = modelFolderPath
        self.isPlotEnable = isPlotEnable

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        logger = PlotLogger(self.isPlotEnable)
        mlModel = LstmAutoencoder(feature_size,4,output_size,2)
        taskName = 'RAETask'
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
