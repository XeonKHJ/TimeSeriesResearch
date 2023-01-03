import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class TwowayRNN(nn.Module):
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
        self.forwardCalculation = nn.Linear(2*hidden_size,output_size)
        self.finalCalculation = nn.Sigmoid()
        self.head_linear = nn.Linear(hidden_size,output_size)
        self.tail_linear = nn.Linear(hidden_size,output_size)
        self.head_final = nn.Sigmoid()
        self.tail_final = nn.Sigmoid()

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, b = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)

        to_rx = torch.flip(to_x, [1])
        rx = torchrnn.pack_padded_sequence(to_rx, xTimestampSizes, True)
        rx, rb = self.rlstm(rx)
        
        x, xBatchSize = torchrnn.pad_packed_sequence(x, batch_first=True)
        rx, rxBatchSize = torchrnn.pad_packed_sequence(rx, batch_first=True)


        xrx = torch.stack([x, rx], 2)
        xrx = torch.transpose(xrx, 2, 3)

        xrx = torch.reshape(xrx, (xrx.shape[0], xrx.shape[1], 2*self.hidden_size))

        x = self.forwardCalculation(xrx)
        x = self.finalCalculation(x)

        return x

    def getInputTensor(self, dataset, datasetLengths):
        inputList = torch.split(dataset, 1, 1)
        inputLengths = (numpy.array(datasetLengths)).tolist()
        outputDataset = torch.zeros([dataset.shape[0], dataset.shape[1], dataset.shape[2]])
        inputDataset = torch.zeros([dataset.shape[0], dataset.shape[1], dataset.shape[2]])
        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                outputDataset[j][i] = 1.0

        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                inputDataset[j][i] = inputList[i][j]
        return inputDataset, outputDataset, inputLengths