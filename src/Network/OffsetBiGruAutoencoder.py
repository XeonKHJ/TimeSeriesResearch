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
        - ahead: use x data to predict next x data
    """
    def __init__(self, feature_size, hidden_size=1, output_size=1, num_layers=1, ahead=1):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.aheadCount = ahead

        self.flstmEncoder = nn.GRU(feature_size, hidden_size, num_layers,batch_first =True) # utilize the LSTM model in torch.nn 
        self.rlstmEncoder = nn.GRU(feature_size, hidden_size, num_layers,batch_first =True)
        self.lstmDecoder = nn.GRU(2*hidden_size, hidden_size, num_layers, batch_first=True) 
        
        self.forwardCalculation = nn.Linear(hidden_size,output_size)
        self.finalCalculation = nn.Sigmoid()
        self.isCudaSupported = torch.cuda.is_available()

    def forward(self, to_x, xTimestampSizes):
        rTo_x = to_x.flip(1)
        packedX = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        rPackedX =torchrnn.pack_padded_sequence(rTo_x, xTimestampSizes, True)
        fPackedX, fb = self.flstmEncoder(packedX)
        rPackedX, rb = self.rlstmEncoder(rPackedX)
        fPadedX, _ = torchrnn.pad_packed_sequence(fPackedX, batch_first=True)
        rPadedX, _ = torchrnn.pad_packed_sequence(rPackedX, batch_first=True)
        rPadedX = rPadedX.flip(1)
        fPadedX[:, 0:fPadedX.shape[1]-1, :] = fPadedX[:, 1:fPadedX.shape[1], :]
        rPadedX[:, 1:rPadedX.shape[1], :] = rPadedX[:, 0:rPadedX.shape[1]-1, :]
        fPadedX[:, fPadedX.shape[1]-1, :] = 0
        rPadedX[:, 0, :] = 0
        padedX = torch.cat((fPadedX, rPadedX), 2)
        packedX = torchrnn.pack_padded_sequence(padedX, xTimestampSizes, True)
        packedX, b = self.lstmDecoder(packedX)
        padX, lengths = torchrnn.pad_packed_sequence(packedX, batch_first=True)

        padX = self.forwardCalculation(padX)
        # padX = self.finalCalculation(padX)

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