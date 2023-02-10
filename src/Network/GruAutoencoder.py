import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class GruAutoencoder(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, feature_size, hidden_size=1, output_size=1, num_layers=1, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gruEncoder = nn.GRU(feature_size, hidden_size, num_layers,batch_first =True, dropout=dropout) # utilize the LSTM model in torch.nn 
        self.encodeFc = nn.Linear(hidden_size, hidden_size)
        self.gruDecoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout) 
        
        self.forwardCalculation = nn.Linear(hidden_size,output_size)
        # self.finalCalculation = nn.Sigmoid()
        self.isCudaSupported = torch.cuda.is_available()

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, b = self.gruEncoder(x)  # _x is input, size (seq_len, batch, input_size)
        paddedX, length = torchrnn.pad_packed_sequence(x, True)
        paddedX = self.encodeFc(paddedX)
        packedPaddedX = torchrnn.pack_padded_sequence(paddedX, xTimestampSizes, True)
        x, b = self.gruDecoder(packedPaddedX)
        
        x, lengths = torchrnn.pad_packed_sequence(x, batch_first=True)
    
        x = self.forwardCalculation(x)

        return x