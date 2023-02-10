from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.GruAEConfig import GruAEConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class GruAENABArtiExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "GruAENABArti"

    def getExperimentConfig(self):
        dataReader = NABFoldersReader("../../NAB/", "artificial")
        displayDataReader = NABFoldersReader("../../NAB/", "artificial")
        config = GruAEConfig(globalConfig.getModelPath(), self.logger, self.getName())
        trainer = config.getConfig()
        processers = [
            SlidingWindowStepDataProcessor(windowSize=100, step=20),
            ShuffleDataProcessor()
        ]
        datasetSeperator = NoSepDataSeperator()        
        dataNormalizer = DataNormalizer()
        return trainer, dataReader, processers, datasetSeperator, dataNormalizer