import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class GruOneDEncodedAutoencoder(nn.Module):
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
        self.reserveLengthForDecode = 200

        self.encodedFeatureSize = 20

        self.lstmEncoder = nn.GRU(feature_size, self.encodedFeatureSize, num_layers,batch_first =True, dropout=0.2) # utilize the LSTM model in torch.nn 
        self.lstmEncoderForward = nn.Linear(self.encodedFeatureSize, self.encodedFeatureSize)
        # self.relu = nn.ReLU()
        self.lstmDecoder = nn.GRU(self.encodedFeatureSize, hidden_size, num_layers, batch_first=True) 
        
        self.forwardCalculation = nn.Linear(hidden_size,output_size)
        self.finalCalculation = nn.Sigmoid()

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, hiddenOutputX = self.lstmEncoder(x)  # _x is input, size (seq_len, batch, input_size)
        paddedX, paddedXLengthes = torchrnn.pad_packed_sequence(x, True)
        paddedX[:,0:self.reserveLengthForDecode, :] = paddedX[:,paddedX.shape[1] - self.reserveLengthForDecode:paddedX.shape[1] ,:]
        paddedX[:, self.reserveLengthForDecode:paddedX.shape[1], :] = 0
        paddedX = self.lstmEncoderForward(paddedX)
        packedX = torchrnn.pack_padded_sequence(paddedX, xTimestampSizes, True)
        x, b = self.lstmDecoder(packedX)

        x, lengths = torchrnn.pad_packed_sequence(x, batch_first=True)

        x = self.forwardCalculation(x)
        x = self.finalCalculation(x)

        return x

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