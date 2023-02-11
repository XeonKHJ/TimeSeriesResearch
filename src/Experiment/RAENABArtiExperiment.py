from DataNormalizer.DataNormalizer import DataNormalizer
from DataProcessor.PartitionDataProcessor import PartitionDataProcessor
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig
from TaskConfig.RAETaskConfig import RAETaskConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

# RAE on NAB artifical dataset.
class RAENABArtiExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "RAENABArti"

    def getExperimentConfig(self):
        dataReader = NABFoldersReader("../../NAB/", "artificial")
        validDataReader = NABFoldersReader("../../NAB/", "artificial")
        config = RAETaskConfig(self.logger, self.getName(), showTrainningInfo=True)
        trainer = config.getConfig()
        processers = [
            PartitionDataProcessor(0.5),
            ShuffleDataProcessor()
        ]
        datasetSeperator = NoSepDataSeperator()        
        dataNormalizer = DataNormalizer()
        return trainer, dataReader, validDataReader, processers, datasetSeperator, dataNormalizer