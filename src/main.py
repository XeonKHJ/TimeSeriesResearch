import torch
import torch.nn
import numpy
import matplotlib.pyplot
import math
from DatasetReader.SmallNABReader import SmallNABReader
from Network.LstmAutoencoder import LstmAutoencoder
from Network.OffsetTwowayRNN import OffsetTwowayRNN
from Network.TraditionLstm import TraditionLstm
from DatasetReader.NABReader import NABReader
from Network.TwowayRNN import TwowayRNN

folder = "C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\realTweets\\realTweets"
datasetReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\realTweets\\realTweets")

normalDataReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\artificialNoAnomaly\\artificialNoAnomaly")
abnormalDataReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\artificialWithAnomaly\\artificialWithAnomaly")

if __name__ == '__main__':
    # dataset, datasetLengths = datasetReader.read()

    normalDataset, normalDatasetLengths = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths = abnormalDataReader.read()

    feature_size = 1
    output_size = 1


    lstm_model = LstmAutoencoder(feature_size,4,output_size,2)

    maxData = max(normalDataset.max(), abnormalDataset.max())
    minData = min(normalDataset.min(), abnormalDataset.min())
    normalDataset = (normalDataset - minData) / (maxData - minData)
    abnormalDataset = (abnormalDataset - minData) / (maxData - minData)

    seprateIdx = math.ceil(normalDataset.__len__()/2)

    trainDataset = normalDataset[0:seprateIdx]
    trainsetLengths = normalDatasetLengths[0:seprateIdx]

    toTrainDataset, labelDataset, labelDatasetLengths = lstm_model.getInputTensor(trainDataset, trainsetLengths)

    validDataset = normalDataset[seprateIdx:normalDataset.__len__()]
    validsetLengths = normalDatasetLengths[seprateIdx:normalDataset.__len__()]

    dataset = normalDataset
    datasetLengths = normalDatasetLengths
    
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
    lossFunc = torch.nn.MSELoss()

    batchSize = 1
    currentIdx = 0
    datasetSize = toTrainDataset.shape[0]
    epoch = 0
    
    
    while True:
        startIdx = currentIdx % (datasetSize - batchSize)
        endIdx = startIdx + batchSize
        currentIdx += batchSize 
        # print("startIdx\t", startIdx, "\tendIdx\t", endIdx)
        trainSet = toTrainDataset[startIdx:endIdx]
        output = lstm_model(trainSet, labelDatasetLengths[startIdx:endIdx])
        labelTensor = torch.full(output.shape, 1.0)
        loss = lossFunc(output, labelDataset[startIdx:endIdx])
        # print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch += 1
        if epoch % 10 == 0:
            # analysisLossFunc = torch.nn.MSELoss()
            validInput, validOutput, validLengthes = lstm_model.getInputTensor(validDataset, validsetLengths)
            abInput, abOutput, abLengths = lstm_model.getInputTensor(abnormalDataset, abnormalDatasetLengths)
            anaOutput = lstm_model(validInput, validLengthes)
            # normalResult = analysisLossFunc(anaOutput, torch.full(anaOutput.shape, 1.0))
            anaAbnormalOutput = lstm_model(abInput, abLengths)
            # abnormalResult = analysisLossFunc(anaAbnormalOutput, torch.full(anaAbnormalOutput.shape, 0.0))
            print("result\t", torch.mean(anaOutput).item(), "\t", torch.mean(anaAbnormalOutput).item(), "\t", loss.item())
            
            x = validOutput[1].reshape([-1]).tolist()
            px = anaOutput[1].reshape([-1]).tolist()

            abx = abOutput[0].reshape([-1]).tolist()
            abpx = anaAbnormalOutput[0].reshape([-1]).tolist()

            fig, ax = matplotlib.pyplot.subplots()
            fig2, ax2 = matplotlib.pyplot.subplots()
            print("ready to draw")
            ax.plot(x, label="dataset")
            ax.plot(px, label="predict")
            ax.legend()
            ax2.plot(abx, label="ab dataset")
            ax2.plot(abpx, label="ab predict")
            ax2.legend()
            matplotlib.pyplot.show()