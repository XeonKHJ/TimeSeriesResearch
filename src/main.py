import torch
import torch.nn
import numpy
from DatasetReader.SmallNABReader import SmallNABReader
from TwowayRNN import LstmRNN
from DatasetReader.NABReader import NABReader

folder = "C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\realTweets\\realTweets"
datasetReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\realTweets\\realTweets")

normalDataReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\artificialNoAnomaly\\artificialNoAnomaly")
abnormalDataReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\artificialWithAnomaly\\artificialWithAnomaly")

if __name__ == '__main__':
    # dataset, datasetLengths = datasetReader.read()

    normalDataset, normalDatasetLengths = normalDataReader.read()
    abnormalDataset, abnormalDatasetLengths = abnormalDataReader.read()

    maxData = max(normalDataset.max(), abnormalDataset.max())
    normalDataset = normalDataset / maxData
    abnormalDataset = abnormalDataset / maxData

    trainDatset = normalDataset[0:round(normalDataset.__len__()/2)]
    trainsetLengths = normalDatasetLengths[0:round(normalDataset.__len__()/2)]
    validDataset = normalDataset[round(normalDataset.__len__()/2):normalDataset.__len__()]
    validsetLengths = normalDatasetLengths[round(normalDataset.__len__()/2):normalDataset.__len__()]

    dataset = normalDataset
    datasetLengths = normalDatasetLengths

    feature_size = 1
    output_size = 1
    lstm_model = LstmRNN(feature_size,4,output_size,4)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
    lossFunc = torch.nn.MSELoss()

    batchSize = 2
    currentIdx = 0
    datasetSize = dataset.__len__()
    epoch = 0
    
    while True:
        startIdx = currentIdx % (datasetSize - batchSize)
        endIdx = startIdx + batchSize
        currentIdx += batchSize 
        # print("startIdx\t", startIdx, "\tendIdx\t", endIdx)
        trainSet = dataset[startIdx:endIdx]
        output = lstm_model(trainSet, datasetLengths[startIdx:endIdx])
        labelTensor = torch.full(output.shape, 1.0)
        loss = lossFunc(output, labelTensor)
        # print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch += 1
        if epoch % 2 == 0:
            # analysisLossFunc = torch.nn.MSELoss()
            anaOutput = lstm_model(validDataset, validsetLengths)
            # normalResult = analysisLossFunc(anaOutput, torch.full(anaOutput.shape, 1.0))
            anaAbnormalOutput = lstm_model(abnormalDataset, abnormalDatasetLengths)
            # abnormalResult = analysisLossFunc(anaAbnormalOutput, torch.full(anaAbnormalOutput.shape, 0.0))
            print("result\t", torch.mean(anaOutput).item(), "\t", torch.mean(anaAbnormalOutput).item(), "\t", loss.item())
            