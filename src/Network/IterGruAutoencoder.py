import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

# decode part allows pass output from previous rnn unit to next rnn unit.
class IterGruAutoencoder(nn.Module):
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

        self.gruEncoder = nn.GRU(feature_size, hidden_size, num_layers,batch_first =True) # utilize the LSTM model in torch.nn 
        self.encodeFc = nn.Linear(hidden_size, hidden_size)
        self.gruDecoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True) 
        
        self.forwardCalculation = nn.Linear(hidden_size,output_size)
        # self.finalCalculation = nn.Sigmoid()
        self.isCudaSupported = torch.cuda.is_available()

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, hiddenOutput = self.gruEncoder(x)  # _x is input, size (seq_len, batch, input_size)
        paddedX, _ = torchrnn.pad_packed_sequence(x, True)
        encoded = hiddenOutput[hiddenOutput.shape[0]-1, :, :].reshape([hiddenOutput.shape[1], 1, -1])
        encoded = self.encodeFc(encoded)
        paddedX[:,0,:] = paddedX[:,paddedX.shape[1]-1, :]
        paddedX[:,1:to_x.shape[1],:] = 0
        # paddedX = self.encodeFc(paddedX)

        # decoderOutput = torch.zeros([to_x.shape[0], to_x.shape[1], self.hidden_size], device=torch.device('cuda'), requires_grad=True)
        encoded, encodedH = self.gruDecoder(encoded, hiddenOutput)
        decoderOutput = encoded
        for idx in range(1, to_x.shape[1]):
            decoderOutputUnit, encodedH = self.gruDecoder(decoderOutput[:, idx-1, :].reshape(paddedX.shape[0], 1, -1), encodedH)
            decoderOutput = torch.cat((decoderOutput, decoderOutputUnit), 1)
        x = self.forwardCalculation(decoderOutput)
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

        if self.isCudaSupported:
            return inputDataset.cuda(), outputDataset.cuda(), inputLengths
        else:
            return inputDataset, outputDataset, inputLengths