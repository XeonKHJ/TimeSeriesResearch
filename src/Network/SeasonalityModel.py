import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy


class SeasonalityModel(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, feature_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # Seasonality in paper Forecasting at scale
        self.seasonalityN = 3
        seasonalityOutputFeatureSize = self.seasonalityN * 2 + 1 + 1 + 1
        self.lstmSeasonality = nn.LSTM(
            input_size=feature_size, hidden_size=seasonalityOutputFeatureSize, num_layers=num_layers, batch_first=True)

        self.forwardCalculation = nn.Linear(hidden_size, 1)
        self.finalCalculation = nn.Sigmoid()

        self.seasonality = torch.zeros([0])

    def forward(self, to_x, xTimestampSizes):
        if self.seasonality.shape == torch.Size([0]):
            self.inputShape = to_x.shape[1]
            # self.seasonality = torch.rand([to_x.shape[0], to_x.shape[1]])

        packedX = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        packedSeasonalityX, b = self.lstmSeasonality(packedX)
        paddedSeasonalityX, seasonalityXLengthes = torchrnn.pad_packed_sequence(
            packedSeasonalityX, batch_first=True)
        seasonalityMatrix = 12
        seasonalityX = paddedSeasonalityX[:,
                                          paddedSeasonalityX.shape[1] - 1, :]
        # seasonalityABMatrix = [a1,a2,a3,...,aN,b1,b2,...,bN]
        seasonalityABMatrix = seasonalityX[:, 0:2 * self.seasonalityN]
        seasonalityPs = seasonalityX[:,self.seasonalityN]
        seasonalityPs = seasonalityPs.reshape([seasonalityPs.shape[0],1])
        seasonalityPs = seasonalityPs.repeat(1,self.seasonalityN)
        seasonalityNs = torch.arange(1, self.seasonalityN + 1, 1) * 2 * torch.pi
        seasonalityNs = seasonalityNs.repeat(paddedSeasonalityX.shape[0], 1)
        seasonalitySequence = seasonalityNs * seasonalityPs

        outputX = torch.zeros(to_x.shape)
        for i in range(paddedSeasonalityX.shape[1]):
            currentT = i + 1
            twopipt = currentT * seasonalitySequence
            coses = torch.cos(twopipt)
            sins = torch.sin(twopipt)
            cossinMatrix = torch.concat((coses,sins), 1)
            abcossinMatrix = seasonalityABMatrix * cossinMatrix
            outputX[:,i] = seasonalityX[:,2 * self.seasonalityN + 2].reshape([-1,1]) * abcossinMatrix.sum(1).reshape([-1,1]) + seasonalityX[:,2 * self.seasonalityN + 1].reshape([-1,1])
        
        x = outputX
        return x

    def getInputTensor(self, dataset, datasetLengths):
        inputList = torch.split(dataset, 1, 1)
        inputLengths = (numpy.array(datasetLengths)).tolist()
        outputDataset = torch.zeros(
            [dataset.shape[0], dataset.shape[1], dataset.shape[2]])
        inputDataset = torch.zeros(
            [dataset.shape[0], dataset.shape[1], dataset.shape[2]])
        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                outputDataset[j][i] = inputList[i][j]

        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                inputDataset[j][i] = inputList[i][j]
        return inputDataset, outputDataset, inputLengths
