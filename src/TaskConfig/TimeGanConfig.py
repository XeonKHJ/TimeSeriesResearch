from Network.GruEncoder import GruEncoder
from Network.GruOneDEncoder import GruOneDEncoder
from Network.GruOneDEncodedAutoencoder import GruOneDEncodedAutoencoder
from Logger.PlotLogger import PlotLogger
from Network.BiGruAutoencoder import BiGruAutoencoder
from TaskConfig.ITaskConfig import ITaskConfig
from Trainers.RAETrainer import RAETrainer
from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

import torch
import os.path as path

from Trainers.RandomRAETrainer import RandomRAETrainer
from Trainers.TimeGANTrainer import TimeGANTrianer

class TimeGanConfig(ITaskConfig):
    def __init__(self, modelFolderPath, isPlotEnable=False):
        self.modelFolderPath = modelFolderPath
        self.isPlotEnable = isPlotEnable

    def getConfig(self, isCuda = False):
        feature_size = 1
        staticFeatSize = 20
        tempFeatSize = 10
        output_size = 1
        staticFeature = GruOneDEncoder(1, 4, staticFeatSize, 2)
        tempFeature = GruEncoder(feature_size, tempFeatSize, 2)
        generator = BiGruAutoencoder(staticFeatSize+feature_size+tempFeatSize,4,output_size,output_size)
        discriminator = BiGruAutoencoder(output_size,4,output_size,1)
        
        logger = PlotLogger(self.isPlotEnable)
        taskName = 'TimeGanConfig'
        try:
            generator.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + '-generator' + ".pt")))
            discriminator.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + '-discriminator' + ".pt")))
            staticFeature.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + '-staticFeat' + ".pt")))
            staticFeature.load_state_dict(torch.load(path.join(self.modelFolderPath, taskName + '-tempFeat' + ".pt")))
        except:
            pass
        if torch.cuda.is_available():
            generator.cuda()
            discriminator.cuda()
            staticFeature.cuda()
            tempFeature.cuda()
        trainer = TimeGANTrianer(generator, discriminator, staticFeature, tempFeature, logger, taskName, 1e-9)
        datasetSeperator = NoSepDataSeperator()
        # logger = PlotLogger()
        
        dataNormalizer = DataNormalizer()
        
        return generator, datasetSeperator, trainer, logger, dataNormalizer, taskName
