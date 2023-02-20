from DataNormalizer.DataNormalizer import DataNormalizer
from DataNormalizer.StandardScalerDataNormalizer import StandardScalerDataNormalizer
from DataProcessor.LabelOffsetDataProcessor import LabelOffsetDataProcessor
from DataProcessor.PartitionDataProcessor import PartitionDataProcessor
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DatasetReader.NABFileReader import NABFileReader
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.AheadTaskConfig import AheadTaskConfig
from TaskConfig.AheadWithErrorTaskConfig import AheadWithErrorTaskConfig
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class AheadWithErrorGruAEArtiExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "AheadWithErrorGruAEArti"

    def getExperimentConfig(self):
        normalDataReader = NABFileReader("../../NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")
        dataReader = NABFileReader("../../NAB/", "artificialWithAnomaly/art_daily_jumpsdown.csv")
        config = AheadWithErrorTaskConfig(self.logger, self.getName(), showTrainingInfo=True)
        trainer = config.getConfig()
        windowSize = 200
        processers = [
            # LabelOffsetDataProcessor(windowSize),
            # PartitionDataProcessor(0.5),
            SlidingWindowStepDataProcessor(windowSize=windowSize, step=1),
            ShuffleDataProcessor()
        ]
        datasetSeperator = NoSepDataSeperator()        
        dataNormalizer = DataNormalizer()
        return trainer, normalDataReader, dataReader, processers, datasetSeperator, dataNormalizer