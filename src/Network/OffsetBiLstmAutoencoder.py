import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class OffsetBiLstmAutoencoder(nn.Module):
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

        self.lstmEncoder = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.rlstmEncoder = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)

        self.lstmDecoder = nn.LSTM(2 * hidden_size, hidden_size, num_layers,batch_first =True)

        self.forwardCalculation = nn.Linear(hidden_size,output_size)
        self.finalCalculation = nn.Sigmoid()

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        encoded_packed_x, b = self.lstmEncoder(x)
        encoded_packed_rx, b = self.rlstmEncoder(x)

        encoded_paded_x, xBatchSize = torchrnn.pad_packed_sequence(encoded_packed_x, True)
        encoded_paded_rx, rxBatchSize = torchrnn.pad_packed_sequence(encoded_packed_rx, True)

        encoded_paded_xrx = torch.concat((encoded_paded_rx, encoded_paded_x), 2)
        encoded_packed_xrx = torchrnn.pack_padded_sequence(encoded_paded_xrx, xTimestampSizes, True)

        decoded_packed_x, b = self.lstmDecoder(encoded_packed_xrx)
        decoded_paded_x, decodedXBatchSize = torchrnn.pad_packed_sequence(decoded_packed_x, True)

        x = self.forwardCalculation(decoded_paded_x)
        x = self.finalCalculation(x)

        return x

    def getInputTensor(self, dataset, datasetLengths):
        inputList = torch.split(dataset, 1, 1)
        inputLengths = (numpy.array(datasetLengths) - 2).tolist()
        outputDataset = torch.zeros([dataset.shape[0], dataset.shape[1] - 2, dataset.shape[2]])
        inputDataset = torch.zeros([dataset.shape[0], dataset.shape[1], dataset.shape[2]])
        for i in range(inputList.__len__() - 2):
            for j in range(outputDataset.shape[0]):
                outputDataset[j][i] = inputList[i+1][j]

        for i in range(inputList.__len__()):
            for j in range(outputDataset.shape[0]):
                inputDataset[j][i] = inputList[i][j]
        return inputDataset, outputDataset, inputLengths