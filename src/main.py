import torch
import torch.nn
import os.path as path


from TaskConfig.RAETaskConfig import RAETaskConfig
from DatasetReader.NABReader import NABReader
import ArgParser

normalDataReader = NABReader("../datasets/preprocessed/NAB/artificialNoAnomaly/artificialNoAnomaly")
abnormalDataReader = NABReader("../datasets/preprocessed/NAB/artificialWithAnomaly/artificialWithAnomaly")
# skabDataReader = SKABDatasetReader("C:\\Users\\redal\\source\\repos\\SKAB\\data\\valve1")
modelFolderPath = "SavedModels"

config = RAETaskConfig(modelFolderPath)
def getConfig():
    mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName = config.getConfig()
    return mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName

def logEvalModel(mlModel):
    mlModel.eval()
    normalIdx = 3
    abnormalIdx = 5
    validInput, validOutput, validLengthes = mlModel.getInputTensor(validDataset, validsetLengths)
    abInput, abOutput, abLengths = mlModel.getInputTensor(abnormalDataset, abnormalDatasetLengths)
    anaOutput = mlModel(validInput, validLengthes)
    anaAbnormalOutput = mlModel(abInput, abLengths)
    # print("result\t", torch.mean(anaOutput).item(), "\t", torch.mean(anaAbnormalOutput).item(), "\t", loss.item())
    x = validOutput[normalIdx].reshape([-1]).tolist()
    px = anaOutput[normalIdx].reshape([-1]).tolist()

    tl = anaAbnormalOutput[abnormalIdx]
    t = abOutput[abnormalIdx]
    ts = t - tl
    tlList = tl.reshape([-1]).tolist()
    tList = t.reshape([-1]).tolist()
    tsList = ts.reshape([-1]).tolist()
    maxDiff = (torch.abs(validOutput - anaOutput)).max().item()
    print("max diff\t", maxDiff)
    logger.logResults([tList, tlList, tsList], ["t", "tl", "ts"])

if __name__ == '__main__':

    args = ArgParser.getArgs()
    isLoggerEnable = not (args.disablePlot)

    # dataset, datasetLengths = datasetReader.read()
    normalDataset, normalDatasetLengths = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths = abnormalDataReader.read()
    # skabDataReader, skabDataLengths = skabDataReader.read()
    

    mlModel, datasetSeperator, trainer, logger, dataNormalizer, taskName = getConfig()
    
    dataNormalizer.addDatasetToRef(normalDataset)
    dataNormalizer.addDatasetToRef(abnormalDataset)
    normalDataset = dataNormalizer.normalizeDataset(normalDataset)
    abnormalDataset = dataNormalizer.normalizeDataset(abnormalDataset)
    trainDataset = datasetSeperator.getTrainningSet(normalDataset)
    trainsetLengths = datasetSeperator.getTrainningSet(normalDatasetLengths)

    validDataset = datasetSeperator.getValidationSet(normalDataset)
    validsetLengths = datasetSeperator.getValidationSet(normalDatasetLengths)

    toTrainDataset, labelDataset, labelDatasetLengths = mlModel.getInputTensor(trainDataset, trainsetLengths)
    mlModel.train()
    batchSize = toTrainDataset.shape[0]
    currentIdx = 0
    datasetSize = toTrainDataset.shape[0]
    epoch = 0
    
    while True:
        if datasetSize - batchSize == 0:
            startIdx = 0
        else:
            startIdx = currentIdx % (datasetSize - batchSize)
        endIdx = startIdx + batchSize
        currentIdx += batchSize 
        trainSet = toTrainDataset[startIdx:endIdx]
        labelSet = labelDataset[startIdx:endIdx]
        labelSetLengths = labelDatasetLengths[startIdx:endIdx]
        loss = trainer.train(trainSet, labelSetLengths, labelSet)

        if epoch % 100 == 0:
            if isLoggerEnable:
                logEvalModel(mlModel=mlModel)
            torch.save(mlModel.state_dict(), path.join(modelFolderPath, taskName + ".pt"))
            mlModel.train()
        epoch += 1

