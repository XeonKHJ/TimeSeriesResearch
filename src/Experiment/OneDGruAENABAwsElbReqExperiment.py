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

class OneDGruAENABAwsElbReqExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "OneDGruAENABAwsElbReq"

    def getExperimentConfig(self):
        dataReader = NABFilesReader("../../NAB/", "realAWSCloudwatch", "elb_request_count")
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