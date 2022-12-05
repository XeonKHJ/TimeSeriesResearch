import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class LstmRNN(nn.Module):
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
        forward_to_stack_x = torch.transpose(x, 0, 1)
        backward_to_stack_x = torch.transpose(rx, 0, 1)
        # stack x and rx
        T = x.shape[1] - 1

        if T == 0:
            # No need to clap
            head_x = x
        elif T == 1:
            head_x = x
            tail_x = rx
            # No need to clap, but requires reversed LSTM
        elif T > 1:
            forward_stacking_x = torch.transpose(forward_to_stack_x[0:T - 1], 0, 1)
            backward_stacking_x = torch.transpose(backward_to_stack_x[1:T], 0, 1)
            head_x = backward_to_stack_x[1]
            tail_x = forward_to_stack_x[T-1]
            xrx = torch.stack([forward_stacking_x, backward_stacking_x], 2)
            xrx = torch.transpose(xrx, 2, 3)

            xrx = torch.reshape(xrx, (xrx.shape[0], xrx.shape[1], 2*self.hidden_size))

            x = self.forwardCalculation(xrx)
            x = self.finalCalculation(x)
            
            #torch.reshape(head_x, (head_x.shape[0], head_x.shape[1], 1))
            head_x = self.head_linear(head_x)
            head_x = self.head_final(head_x)
            head_x = torch.reshape(head_x, (head_x.shape[0], 1, self.output_size))
            tail_x = self.tail_linear(tail_x)
            tail_x = self.tail_final(tail_x)
            tail_x = torch.reshape(tail_x, (tail_x.shape[0], 1, self.output_size))
            # need to stack forward LSTM and reversed LSTM together.

            stacked_x = torch.concat([head_x, x, tail_x], 1)

        #s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        #x = x.view(s*b, h)

        # stack output 


        #x = torch.reshape(x, (x.shape[0],x.shape[1],x.shape[2]))
        #x = x.view(s, b, -1)
        return stacked_x
    
    
    
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