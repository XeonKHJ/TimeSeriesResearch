import torch
import torch.nn
from DatasetReader.SmallNABReader import SmallNABReader
from TwowayRNN import LstmRNN
from DatasetReader.NABReader import NABReader

datasetReader = SmallNABReader("C:\\Users\\redal\\source\\repos\\TimeSeriesResearch\\datasets\\preprocessed\\NAB\\realTweets\\realTweets")

if __name__ == '__main__':
    dataset, datasetLengths = datasetReader.read()
    dataset = dataset / dataset.max()
    
    feature_size = 1
    lstm_model = LstmRNN(feature_size,4,1,4)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)
    lossFunc = torch.nn.MSELoss()
    while True:
        output = lstm_model(dataset, datasetLengths)
        loss = lossFunc(output, dataset)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()