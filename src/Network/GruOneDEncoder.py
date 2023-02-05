import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

# Encoder only!
class GruOneDEncoder(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, feature_size, hidden_size=1, output_size=1, num_layers=1, reserveLengthForDecode=1):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.reserveLengthForDecode = reserveLengthForDecode

        self.encodedFeatureSize = 20

        self.lstmEncoder = nn.GRU(feature_size, self.encodedFeatureSize, num_layers,batch_first =True, bidirectional=True) # utilize the LSTM model in torch.nn 
        self.lstmEncoderForward = nn.Linear(self.encodedFeatureSize * 2, self.encodedFeatureSize)

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, hiddenOutputX = self.lstmEncoder(x)  # _x is input, size (seq_len, batch, input_size)
        paddedX, paddedXLengthes = torchrnn.pad_packed_sequence(x, True)
        paddedX = hiddenOutputX[self.num_layers * 2 - 2:self.num_layers * 2].transpose(1,0).reshape(to_x.shape[0], 1, -1)
        paddedX = self.lstmEncoderForward(paddedX)
        return paddedX

    def getInputTensor(self, dataset, datasetLengths):
        inputList = torch.split(dataset, 1, 1)
        inputLengths = (numpy.array(datasetLengths)).tolist()
        outputDataset = torch.zeros([dataset.shape[0], dataset.shape[1] , dataset.shape[2]])
        inputDataset = torch.zeros([dataset.shape[0], dataset.shape[1], dataset.shape[2]])
        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                outputDataset[j][i] = inputList[i][j]

        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                inputDataset[j][i] = inputList[i][j]
        if torch.cuda.is_available():
            return inputDataset.cuda(), outputDataset.cuda(), inputLengths
        else:
            return inputDataset, outputDataset, inputLengths