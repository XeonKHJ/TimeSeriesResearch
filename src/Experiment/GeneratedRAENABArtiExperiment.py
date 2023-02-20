from DataNormalizer.DataNormalizer import DataNormalizer
from DataProcessor.PartitionDataProcessor import PartitionDataProcessor
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.GeneratedTaskConfig import GeneratedTaskConfig
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig
from TaskConfig.RAETaskConfig import RAETaskConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

# RAE on NAB artifical dataset.
class GeneratedRAENABArtiExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "GeneratedRAENABArti"

    def getExperimentConfig(self):
        dataReader = NABFoldersReader("../../NAB/", "artificial")
        validDataReader = NABFoldersReader("../../NAB/", "artificial")
        config = GeneratedTaskConfig(self.logger, self.getName(), showTrainningInfo=True)
        trainer = config.getConfig()
        processers = [
            SlidingWindowStepDataProcessor(windowSize=100, step=20),
            ShuffleDataProcessor()
        ]
        datasetSeperator = NoSepDataSeperator()        
        dataNormalizer = DataNormalizer()
        return trainer, dataReader, validDataReader, processers, datasetSeperator, dataNormalizer