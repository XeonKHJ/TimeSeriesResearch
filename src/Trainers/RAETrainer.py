from Trainers.ITrainer import ITrainer

class RAETrainer(ITrainer):
    def __init__(self):
        super().__init__()

    def train(self, trainSet, labelSet):
        tl = t - ts
        if datasetSize - batchSize == 0:
            startIdx = 0
        else:
            startIdx = currentIdx % (datasetSize - batchSize)
        endIdx = startIdx + batchSize
        currentIdx += batchSize 
        trainSet = toTrainDataset[startIdx:endIdx]
        output = mlModel(trainSet, labelSet)
        labelTensor = torch.full(output.shape, 1.0)
        loss = lossFunc(output, labelDataset[startIdx:endIdx])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()