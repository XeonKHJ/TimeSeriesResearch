import torch
import torch.nn
import numpy
import matplotlib.pyplot
import math
from DatasetReader.SmallNABReader import SmallNABReader
from Logger.PlotLogger import PlotLogger
from Network.LstmAutoencoder import LstmAutoencoder
from Network.OffsetTwowayRNN import OffsetTwowayRNN
from Network.TraditionLstm import TraditionLstm
from DatasetReader.NABReader import NABReader
from Network.TwowayRNN import TwowayRNN
from Network.OffsetBiLstmAutoencoder import OffsetBiLstmAutoencoder

from DataSeperator.NoSepDataSeperator import NoSepDataSeperator

normalDataReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\artificialNoAnomaly\\artificialNoAnomaly")
abnormalDataReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\artificialWithAnomaly\\artificialWithAnomaly")
fileName = "lstmautoencoder.pt"

def getConfig():
    feature_size = 1
    output_size = 1

    try:
        mlModel = torch.load(fileName)
    except:
        mlModel = LstmAutoencoder(feature_size,4,output_size,2)
    
    optimizer = torch.optim.Adam(mlModel.parameters(), lr=1e-2)
    lossFunc = torch.nn.MSELoss()
    datasetSeperator = NoSepDataSeperator()
    logger = PlotLogger()
    return mlModel, datasetSeperator, optimizer, lossFunc, logger


if __name__ == '__main__':
    # dataset, datasetLengths = datasetReader.read()
    normalDataset, normalDatasetLengths = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths = abnormalDataReader.read()

    mlModel, datasetSeperator, optimizer, lossFunc, logger = getConfig()

    maxData = max(normalDataset.max(), abnormalDataset.max())
    minData = min(normalDataset.min(), abnormalDataset.min())
    normalDataset = (normalDataset - minData) / (maxData - minData)
    abnormalDataset = (abnormalDataset - minData) / (maxData - minData)

    trainDataset = datasetSeperator.getTrainningSet(normalDataset)
    trainsetLengths = datasetSeperator.getTrainningSet(normalDatasetLengths)

    validDataset = datasetSeperator.getValidationSet(normalDataset)
    validsetLengths = datasetSeperator.getValidationSet(normalDatasetLengths)

    toTrainDataset, labelDataset, labelDatasetLengths = mlModel.getInputTensor(trainDataset, trainsetLengths)
    
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
        output = mlModel(trainSet, labelDatasetLengths[startIdx:endIdx])
        labelTensor = torch.full(output.shape, 1.0)
        loss = lossFunc(output, labelDataset[startIdx:endIdx])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 2 == 0:
            validInput, validOutput, validLengthes = mlModel.getInputTensor(validDataset, validsetLengths)
            abInput, abOutput, abLengths = mlModel.getInputTensor(abnormalDataset, abnormalDatasetLengths)
            anaOutput = mlModel(validInput, validLengthes)
            anaAbnormalOutput = mlModel(abInput, abLengths)
            print("result\t", torch.mean(anaOutput).item(), "\t", torch.mean(anaAbnormalOutput).item(), "\t", loss.item())
            x = validOutput[1].reshape([-1]).tolist()
            px = anaOutput[1].reshape([-1]).tolist()

            abx = abOutput[1].reshape([-1]).tolist()
            abpx = anaAbnormalOutput[1].reshape([-1]).tolist()

            logger.logResult(x, px)
            logger.logResult(abx, abpx)
            torch.save(mlModel, fileName)
        epoch += 1

        