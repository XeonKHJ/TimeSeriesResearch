import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class TraditionLstm(nn.Module):
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
        # forward LSTM
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers,batch_first =True) # utilize the LSTM model in torch.nn 
        
        # reveresd LSTM
        self.rlstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True) 
        self.forwardCalculation = nn.Linear(hidden_size,output_size)
        self.finalCalculation = nn.Sigmoid()
        self.head_linear = nn.Linear(hidden_size,output_size)
        self.tail_linear = nn.Linear(hidden_size,output_size)
        self.head_final = nn.Sigmoid()
        self.tail_final = nn.Sigmoid()

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, b = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        x, xBatchSize = torchrnn.pad_packed_sequence(x, batch_first=True)
        x = self.forwardCalculation(x)
        x = self.finalCalculation(x)
        return x

    def getInputTensor(self, dataset, datasetLengths):
        inputList = torch.split(dataset, 1, 1)
        inputLengths = (numpy.array(datasetLengths)).tolist()
        outputDataset = torch.zeros([dataset.shape[0], dataset.shape[1] - 1, dataset.shape[2]])
        inputDataset = torch.zeros([dataset.shape[0], dataset.shape[1], dataset.shape[2]])
        for i in range(inputList.__len__() - 1):
            for j in range(outputDataset.shape[0]):
                outputDataset[j][i] = inputList[i+1][j]

        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                inputDataset[j][i] = inputList[i][j]
        return inputDataset, outputDataset, inputLengths