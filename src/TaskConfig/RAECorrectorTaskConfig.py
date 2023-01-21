from Logger.PlotLogger import PlotLogger
from Network.LstmAutoencoder import LstmAutoencoder
from TaskConfig.ITaskConfig import ITaskConfig
from TaskConfig.RAETaskConfig import RAETaskConfig
from Trainers.RAECorrectorTrainer import RAECorrectorTrainer
from Trainers.RAETrainer import RAETrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

class RAECorrectorTaskConfig(ITaskConfig):
    def __init__(self, modelFolderPath, isPlotEnable):
        self.modelFolderPath = modelFolderPath
        self.isPlotEnable = isPlotEnable

    def getConfig(self, isCuda = False):
        feature_size = 1
        output_size = 1
        logger = PlotLogger(self.isPlotEnable)
        aeModel = LstmAutoencoder(feature_size,4,output_size,2)
        correctorModel = LstmAutoencoder(2, 4, output_size, 2)
        aeModelName = 'RAETask-raemodel'
        correctorModelName = 'RAECorrector'
        try:
            aeModel.load_state_dict(torch.load(path.join(self.modelFolderPath, aeModelName + ".pt")))
            correctorModel.load_state_dict(torch.load(path.join(self.modelFolderPath, correctorModelName + ".pt")))
        except:
            pass
        if torch.cuda.is_available():
            aeModel.cuda()
            correctorModel.cuda()
        trainer = RAECorrectorTrainer(aeModel=aeModel, correctorModel=correctorModel, logger = logger)
        datasetSeperator = NoSepDataSeperator()
        dataNormalizer = DataNormalizer()
        
        return aeModel, datasetSeperator, trainer, logger, dataNormalizer, aeModelName
