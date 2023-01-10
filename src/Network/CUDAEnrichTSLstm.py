import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim
import numpy

class CUDAEnrichTSLstm(nn.Module):
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
        enrichedSize = 6
        self.lstmEncoder = nn.LSTM(enrichedSize, hidden_size, num_layers,batch_first =True) # utilize the LSTM model in torch.nn 
        self.lstmDecoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True) 
        
        self.forwardCalculation = nn.Linear(hidden_size,enrichedSize)
        # self.finalCalculation = nn.Sigmoid()

    def forward(self, to_x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(to_x, xTimestampSizes, True)
        x, b = self.lstmEncoder(x)  # _x is input, size (seq_len, batch, input_size)
        x, b = self.lstmDecoder(x)
        x, lengths = torchrnn.pad_packed_sequence(x, batch_first=True)
    
        x = self.forwardCalculation(x)
        # x = self.finalCalculation(x)

        return x

    def getInputTensor(self, dataset, datasetLengths):
        enrichedData, enrichedDataLengthes = self.enrichTimeSeries(dataset)
        return enrichedData.cuda(), enrichedData.cuda(), enrichedDataLengthes

    def enrichTimeSeries(self, data):
        gWindowSize = 4
        gStep = int(gWindowSize / 2) 
        curIdx = 0
        # gTimeLengths represents C' in the paper.
        gTimeLengths = int((2 * data.shape[1] - gWindowSize) / gWindowSize)
        # tensor shape [batch size, data length, nor and don]
        gTensor = torch.zeros([data.shape[0], gTimeLengths, 2])

        prenor = None
        for idx in range(gTimeLengths):
            nor = torch.norm(data[:,curIdx:curIdx + gWindowSize,:], dim=1)
            if prenor != None:
                don = prenor - nor
                gTensor[:,idx,0] = nor.reshape([-1])
                gTensor[:,idx,1] = don.reshape([-1])
            curIdx += gStep
            prenor = nor
 
        # step 2
        # f in the paper
        hWindowSize = 2
        hStep = int(hWindowSize / 2)
        hTimeLengths = int((2 * gTimeLengths - hWindowSize) / hWindowSize)
        hTensor = torch.zeros([data.shape[0], hTimeLengths, 6])
        for idx in range(hTimeLengths):
            hTensor[:,idx,0] = torch.mean(gTensor[:,idx:idx+hWindowSize,0], 1)
            hTensor[:,idx,1] = torch.min(gTensor[:,idx:idx+hWindowSize,0], 1).values
            hTensor[:,idx,2] = torch.max(gTensor[:,idx:idx+hWindowSize,0], 1).values
            hTensor[:,idx,3] = torch.mean(gTensor[:,idx:idx+hWindowSize,1], 1)
            hTensor[:,idx,4] = torch.min(gTensor[:,idx:idx+hWindowSize,1], 1).values
            hTensor[:,idx,5] = torch.max(gTensor[:,idx:idx+hWindowSize,1], 1).values
        lengthTensor = torch.zeros([data.shape[0]])
        lengthTensor[:] = hTimeLengths
        return hTensor, lengthTensor.int().tolist()
    
    def getName(self):
        return "EnrichTLstm"