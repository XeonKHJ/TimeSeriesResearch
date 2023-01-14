import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.utils.rnn as torchrnn
import numpy

class GithubGruAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size, n_layers):
        super().__init__()
        dropout=0.5
        encoder_dim = 30
        
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # encoder layers
        self.lstm1 = nn.GRU(input_dim, hidden_dim, n_layers, dropout=dropout,
                             batch_first=True)#, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim, encoder_dim)

        # decoder layers
        self.fc2 = nn.Linear(encoder_dim, hidden_dim)
        self.lstm2 = nn.GRU(hidden_dim, output_size, n_layers, dropout=dropout,
                             batch_first=True)
        self.finalCal = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, to_x, xTimestampSizes):
        batch_size = to_x.size(0)
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        # decode
        lstm_dec, (hidden1, _) = self.lstm1(x)
        lstm_dec, _ = torchrnn.pad_packed_sequence(lstm_dec, batch_first=True)
        dec = self.dropout(lstm_dec)
        dec = F.relu(self.fc1(dec))

        # encode
        enc = F.relu(self.fc2(dec))
        enc = self.dropout(enc)

        packed_enc = torchrnn.pack_padded_sequence(enc, xTimestampSizes, True)
        lstm_enc, (hidden2, _) = self.lstm2(packed_enc)
        padded_lstm_enc, _ = torchrnn.pad_packed_sequence(lstm_enc, True)
        padded_lstm_enc = self.finalCal(padded_lstm_enc)
        padded_lstm_enc = self.sigmoid(padded_lstm_enc)
        return padded_lstm_enc

    def getName(self):
        return "GithubGruAutoencoder"

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
        return inputDataset.cuda(), outputDataset.cuda(), inputLengths
