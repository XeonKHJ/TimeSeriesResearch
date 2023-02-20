import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class LstmAutoencoderWithCorrector(nn.Module):
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

        self.lstmEncoder = nn.LSTM(feature_size, hidden_size, num_layers,batch_first =True) # utilize the LSTM model in torch.nn 
        self.lstmDecoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True) 
        
        self.lstmCorrector = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True) 
        self.correctorFowardCal = nn.Linear(hidden_size,output_size)

        self.forwardCalculation = nn.Linear(hidden_size,1)
        self.finalCalculation = nn.Sigmoid()

    def forward(self, to_x, xTimestampSizes):
        packedX = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        packedEncodedX, b = self.lstmEncoder(packedX)  # _x is input, size (seq_len, batch, input_size)
        packedDncodedX, b = self.lstmDecoder(packedEncodedX)

        packedCorrectX, b = self.lstmCorrector(packedX)
        paddedX, lengths = torchrnn.pad_packed_sequence(packedDncodedX, batch_first=True)
        paddedCorrectX, b = torchrnn.pad_packed_sequence(packedCorrectX, batch_first=True)
        x = self.forwardCalculation(paddedX)
        correctX = self.correctorFowardCal(paddedCorrectX)
        paddedX = x + correctX
        x = self.finalCalculation(paddedX)

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
        return inputDataset, outputDataset, inputLengths