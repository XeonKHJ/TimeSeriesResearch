from DataNormalizer.DataNormalizer import DataNormalizer
from DataNormalizer.PerDataNormalizer import PerDataNormalizer
from DataNormalizer.StandardScalerDataNormalizer import StandardScalerDataNormalizer
from DataProcessor.LabelOffsetDataProcessor import LabelOffsetDataProcessor
from DataProcessor.PartitionDataProcessor import PartitionDataProcessor
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DatasetReader.NABFileReader import NABFileReader
from DatasetReader.NABFilesReader import NABFilesReader
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.AheadTaskConfig import AheadTaskConfig
from TaskConfig.AheadWithErrorTaskConfig import AheadWithErrorTaskConfig
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class AheadWithErrorGruAENATAwsEC2DiskWriteExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "AheadWithErrorGruAENATAwsEC2DiskWrite"

    def getExperimentConfig(self):
        normalDataReader = NABFilesReader("../../NAB/", "realAWSCloudwatch", "ec2_disk_write_bytes")
        # normalDataReader = NABFileReader("../../NAB/", "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv")
        # dataReader = NABFileReader("../../NAB/", "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv")
        config = AheadWithErrorTaskConfig(self.logger, self.getName(), showTrainingInfo=False)
        trainer = config.getConfig()
        windowSize = 200
        processers = [
            # LabelOffsetDataProcessor(windowSize),
            # PartitionDataProcessor(0.5),
            SlidingWindowStepDataProcessor(windowSize=windowSize, step=1),
            ShuffleDataProcessor()
        ]
        datasetSeperator = NoSepDataSeperator()        
        dataNormalizer = PerDataNormalizer()
        return trainer, normalDataReader, normalDataReader, processers, datasetSeperator, dataNormalizer