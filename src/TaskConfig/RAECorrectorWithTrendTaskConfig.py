from Network.GruAutoencoder import GruAutoencoder
from Network.BiGruAutoencoder import BiGruAutoencoder
from Network.GruOneDEncodedAutoencoder import GruOneDEncodedAutoencoder
from Logger.PlotLogger import PlotLogger
from Network.LstmAutoencoder import LstmAutoencoder
from TaskConfig.ITaskConfig import ITaskConfig
from TaskConfig.RAETaskConfig import RAETaskConfig
from Trainers.RAECorrectorTrainer import RAECorrectorTrainer
from Trainers.RAETrainer import RAETrainer
from Trainers.RAECorrectorWithTrendTrainer import RAECorrectorWithTrendTrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class RAECorrectorWithTrendTaskConfig(ITaskConfig):
    def __init__(self, modelFolderPath, isPlotEnable):
        self.modelFolderPath = modelFolderPath
        self.isPlotEnable = isPlotEnable

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        logger = PlotLogger(self.isPlotEnable)
        generatorModel = LstmAutoencoder(feature_size,4,output_size,2)
        trendModel = GruOneDEncodedAutoencoder(feature_size,4,output_size,2)
        correctorModel = GruAutoencoder(2,4,output_size,2)
        generatorModelName = 'RAETask-raemodel'
        correctorModelName = 'RAECorrectorWithTrend'
        trendModelName = 'OneDAutoencoderConfig'
        try:
            generatorModel.load_state_dict(torch.load(path.join(self.modelFolderPath, generatorModelName + ".pt")))
            trendModel.load_state_dict(torch.load(path.join(self.modelFolderPath, trendModelName + ".pt")))
            correctorModel.load_state_dict(torch.load(path.join(self.modelFolderPath, correctorModelName + ".pt")))           
        except:
            pass
        if torch.cuda.is_available():
            generatorModel.cuda()
            correctorModel.cuda()
            trendModel.cuda()
        trainer = RAECorrectorWithTrendTrainer(generatorModel=generatorModel, correctorModel=correctorModel, trendModel=trendModel, logger = logger)
        datasetSeperator = NoSepDataSeperator()
        dataNormalizer = DataNormalizer()
        
        return generatorModel, datasetSeperator, trainer, logger, dataNormalizer, generatorModelName
