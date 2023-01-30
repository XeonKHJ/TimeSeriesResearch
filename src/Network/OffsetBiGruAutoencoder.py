import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class OffsetBiGruAutoencoder(nn.Module):
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

        self.lstmEncoder = nn.GRU(feature_size, hidden_size, num_layers,batch_first =True, bidirectional=True) # utilize the LSTM model in torch.nn 
        self.lstmDecoder = nn.GRU(2*hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True) 
        
        self.forwardCalculation = nn.Linear(2*hidden_size,output_size)
        self.finalCalculation = nn.Sigmoid()
        self.isCudaSupported = torch.cuda.is_available()

    def forward(self, to_x, xTimestampSizes):
        packedX = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        packedX, b = self.lstmEncoder(packedX)
        padX, lengths = torchrnn.pad_packed_sequence(packedX, batch_first=True)
        padX[:, 0:padX.shape[1]-1, 0:self.feature_size] = padX[:, 1:padX.shape[1], 0:self.feature_size]
        padX[:, 1:padX.shape[1], self.feature_size:2*self.feature_size] = padX[:, 0:padX.shape[1]-1, self.feature_size:2*self.feature_size]
        padX[:, padX.shape[1]-1, 0:self.feature_size] = 0
        padX[:, 0, self.feature_size:2*self.feature_size] = 0
        
        packedX = torchrnn.pack_padded_sequence(padX, xTimestampSizes, True)
        packedX, b = self.lstmDecoder(packedX)
        padX, lengths = torchrnn.pad_packed_sequence(packedX, batch_first=True)

        padX = self.forwardCalculation(padX)
        padX = self.finalCalculation(padX)

        return padX

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

        if self.isCudaSupported:
            return inputDataset.cuda(), outputDataset.cuda(), inputLengths
        else:
            return inputDataset, outputDataset, inputLengths

    def getName(self):
        return "LstmAutoencoder"