import torch
import torch.nn
from TwowayRNN import LstmRNN
from DatasetReader.NABReader import NABReader

datasetReader = NABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\realTweets\\realTweets")

if __name__ == '__main__':
    dataset = datasetReader.read()
    
    lstm_model = LstmRNN(4,4,4,4)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
    lossFunc = torch.nn.MSELoss()
    while True:
        output = lstm_model(dataset)
        loss = lossFunc()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()