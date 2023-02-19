from DataNormalizer.DataNormalizer import DataNormalizer
from DataNormalizer.PerDataNormalizer import PerDataNormalizer
from DataProcessor.PartitionDataProcessor import PartitionDataProcessor
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DatasetReader.NABFilesReader import NABFilesReader
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig
from TaskConfig.RAETaskConfig import RAETaskConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

# RAE on NAB artifical dataset.
class RAENABExperiment(object):
    def __init__(self, logger, firstName, secondName):
        self.logger = logger
        self.firstName = firstName
        self.secondName = secondName
        self.experimentName = "RAENAB-" + firstName + "-" + secondName

    def getName(self):
        return self.experimentName

    def getExperimentConfig(self):
        dataReader = NABFilesReader("../../NAB/", self.firstName, self.secondName)
        # validDataReader = NABFoldersReader("../../NAB/", "artificial")
        config = RAETaskConfig(self.logger, self.getName(), showTrainningInfo=True)
        trainer = config.getConfig()
        processers = [
            PartitionDataProcessor(0.5),
            ShuffleDataProcessor()
        ]
        datasetSeperator = NoSepDataSeperator()        
        dataNormalizer = PerDataNormalizer()
        return trainer, dataReader, dataReader, processers, datasetSeperator, dataNormalizer