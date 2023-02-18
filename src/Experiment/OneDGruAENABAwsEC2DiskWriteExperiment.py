from DataNormalizer.DataNormalizer import DataNormalizer
from DataNormalizer.PerDataNormalizer import PerDataNormalizer
from DataSeperator.TrainAndValidateDataSeprator import TrainAndValidateDataSeprator
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DatasetReader.NABFilesReader import NABFilesReader
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class OneDGruAENABAwsEC2DiskWriteExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "OneDGruAENABAwsEC2DiskWrite"

    def getExperimentConfig(self):
        dataReader = NABFilesReader("../../NAB/", "realAWSCloudwatch", "ec2_disk_write")
        # dataReader = NABFoldersReader("../../NAB/", "realAWSCloudwatch")
        displayDataReader = NABFoldersReader("../../NAB/", "ec2_network_in")
        config = OneDAutoencoderConfig(globalConfig.getModelPath(), self.logger, self.getName(), showTrainingInfo=True)
        trainer = config.getConfig()
        processers = [
            SlidingWindowStepDataProcessor(windowSize=100, step=20),
            ShuffleDataProcessor()
        ]
        datasetSeperator = TrainAndValidateDataSeprator(0.5)        
        dataNormalizer = PerDataNormalizer()
        return trainer, dataReader, dataReader, processers, datasetSeperator, dataNormalizer