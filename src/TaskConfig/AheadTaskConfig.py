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
    def __init__(self, modelFolderPath, isLogEnable, inputFeatureCount=1, outputFeatureCount=1, fileList=[]):
        self.modelFolderPath = modelFolderPath
        self.isLogEnable = isLogEnable
        self.inputFeatureSize = inputFeatureCount
        self.outputFeatureSize = outputFeatureCount
        self.fileList = fileList

    def getConfig(self, isCuda = False):
        feature_size = self.inputFeatureSize
        output_size = self.outputFeatureSize
        logger = PlotLogger(isPlotEnable=self.isLogEnable)
        mlModel = BiGruAutoencoder(feature_size,10,output_size,4)
        taskName = 'AheadTaskConfig'
        try:
            mlModel.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + ".pt")))
        except:
            pass
        if torch.cuda.is_available():
            mlModel.cuda()
        # trainer = RAETrainer(mlModel, logger, taskName, 10)
        trainer = AheadTrainer(mlModel, taskName, logger, 1e-3, self.fileList)
        datasetSeperator = NoSepDataSeperator()
        # logger = PlotLogger()
        
        dataNormalizer = DataNormalizer()
        
        return mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName
