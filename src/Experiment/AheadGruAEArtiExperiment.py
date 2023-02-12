from DataNormalizer.DataNormalizer import DataNormalizer
from DataSeperator.NoSepDataSeperator import NoSepDataSeperator
from DatasetReader.NABFileReader import NABFileReader
from DatasetReader.NABFoldersReader import NABFoldersReader
from TaskConfig.AheadTaskConfig import AheadTaskConfig
from TaskConfig.OneDAutoencoderConfig import OneDAutoencoderConfig

from globalConfig import globalConfig

from DataProcessor.ShuffleDataProcessor import ShuffleDataProcessor
from DataProcessor.SlidingWindowStepDataProcessor import SlidingWindowStepDataProcessor

class AheadGruAEArtiExperiment(object):
    def __init__(self, logger):
        self.logger = logger

    def getName(self):
        return "AheadGruAEArti"

    def getExperimentConfig(self):
        normalDataReader = NABFoldersReader("../../NAB/", "artificialNoAnomaly")
        dataReader = NABFoldersReader("../../NAB/", "artificial")
        config = AheadTaskConfig(self.logger, self.getName(), showTrainingInfo=True)
        trainer = config.getConfig()
        processers = [
            ShuffleDataProcessor()
        ]
        datasetSeperator = NoSepDataSeperator()        
        dataNormalizer = DataNormalizer()
        return trainer, normalDataReader, dataReader, processers, datasetSeperator, dataNormalizer