import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class OffsetTwowayRNN(nn.Module):
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

    def forward(self, paddedX, xTimestampSizes):
        packedX = torchrnn.pack_padded_sequence(paddedX, xTimestampSizes, True)
        packedX, b = self.lstm(packedX)  # _x is input, size (seq_len, batch, input_size)

        paddedRX = torch.flip(paddedX, [1])
        packedRX = torchrnn.pack_padded_sequence(paddedRX, xTimestampSizes, True)
        packedRX, rb = self.rlstm(packedRX)
        
        paddedX, xBatchSize = torchrnn.pad_packed_sequence(packedX, batch_first=True)
        paddedRX, rxBatchSize = torchrnn.pad_packed_sequence(packedRX, batch_first=True)
        forward_to_stack_x = torch.transpose(packedX, 0, 1)
        backward_to_stack_x = torch.transpose(paddedRX, 0, 1)

        forward_stacking_x = torch.transpose(forward_to_stack_x[0:forward_to_stack_x.shape[0]-2], 0, 1)
        backward_stacking_x = torch.transpose(backward_to_stack_x[2:forward_to_stack_x.shape[0]], 0, 1)

        xrx = torch.stack([forward_stacking_x, backward_stacking_x], 2)
        xrx = torch.transpose(xrx, 2, 3)

        xrx = torch.reshape(xrx, (xrx.shape[0], xrx.shape[1], 2*self.hidden_size))

        packedX = self.forwardCalculation(xrx)
        packedX = self.finalCalculation(packedX)

        return packedX

    def getInputTensor(self, dataset, datasetLengths):
        inputList = torch.split(dataset, 1, 1)
        inputLengths = (numpy.array(datasetLengths)).tolist()
        outputDataset = torch.zeros([dataset.shape[0], dataset.shape[1] - 2, dataset.shape[2]])
        inputDataset = torch.zeros([dataset.shape[0], dataset.shape[1], dataset.shape[2]])
        for i in range(inputList.__len__() - 2):
            for j in range(outputDataset.shape[0]):
                outputDataset[j][i] = inputList[i+1][j]

        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                inputDataset[j][i] = inputList[i][j]
        return inputDataset, outputDataset, inputLengths