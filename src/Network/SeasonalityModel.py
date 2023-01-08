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
        self.seasonalityN = 10
        seasonalityOutputFeatureSize = 10 * 2 + 1
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
        seasonalityABMatrix = seasonalityX[:, 0:20].reshape([1, 20])
        seasonalityPs = seasonalityX[:,20]
        seasnalityNs = torch.arange(1, self.seasonalityN, 1)
        
        

        # Create a tensor cos(2πnt)/p
        ## Create a time sequence tensor [1,2,3,4,...,datalength]
        sequenceTensor = torch.arange(0.0, paddedSeasonalityX.shape[1], 1.0).reshape([1,-1])
        ## Create 2*π*p
        seasonality2piN = seasonalityX[:,20].reshape([-1,1]) * 2 * torch.pi
        tempx =  torch.mm(seasonality2piN,sequenceTensor)
        cosMatrix = torch.cos(tempx)
        sinMatrix = torch.sin(tempx)
        torch.concat(cosMatrix, sinMatrix)

        torch.mm(seasonalityABMatrix, cosMatrix)
        
        seasonX = sequenceTensor.dot(cossinMatrix)
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
