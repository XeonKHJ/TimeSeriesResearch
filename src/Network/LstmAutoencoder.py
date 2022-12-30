import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class LstmAutoencoder(nn.Module):
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
        self.lstmEncoder = nn.LSTM(feature_size, hidden_size, num_layers,batch_first =True) # utilize the LSTM model in torch.nn 
        
        # reveresd LSTM
        self.lstmDecoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True) 

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, b = self.lstmEncoder(x)  # _x is input, size (seq_len, batch, input_size)
        x, b = self.lstmDecoder(x)

        x = torchrnn.pad_packed_sequence(x, batch_first=True)
    
        x = self.forwardCalculation(x)
        x = self.finalCalculation(x)

        return x
    
    @staticmethod
    def PadData(dataLists, featureSize):
        # Sort data first
        dataLists.sort(key=(lambda elem:len(elem)), reverse=True)
        dataTimestampLengths = list()
        for i in range(len(dataLists)):
            dataTimestampLengths.append(len(dataLists[i]))
        

        # Padding data
        longestSeqLength = len(dataLists[0])
        dataBatchSize = len(dataLists)
        
        inputTensor = torch.zeros(dataBatchSize,longestSeqLength, featureSize).int()
        
        for i in range(dataBatchSize):
            currentTimeSeq = 1
            for j in range(len(dataLists[i])):
                inputTensor[i][j] = torch.tensor(dataLists[i][j])
       

        return inputTensor.float(), dataTimestampLengths

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