from DataNormalizer.DataNormalizer import DataNormalizer
from DataNormalizer.PerDataNormalizer import PerDataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DataSeperator.TrainAndValidateDataSeprator import TrainAndValidateDataSeprator
from DatasetReader.NABFilesReader import NABFilesReader
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.GruAEConfig import GruAEConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class GruAENABAWSExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "GruAENABAws"

    def getExperimentConfig(self):
        dataReader = NABFilesReader("../../NAB/", "realAWSCloudwatch", "ec2_cpu_utilization")
        displayDataReader = NABFoldersReader("../../NAB/", "artificial")
        config = GruAEConfig(globalConfig.getModelPath(), self.logger, self.getName(), showTrainingInfo=False)
        trainer = config.getConfig()
        processers = [
            SlidingWindowStepDataProcessor(windowSize=100, step=20),
            ShuffleDataProcessor()
        ]
        datasetSeperator = TrainAndValidateDataSeprator(0.5)        
        dataNormalizer = PerDataNormalizer()
        return trainer, dataReader,dataReader, processers, datasetSeperator, dataNormalizer