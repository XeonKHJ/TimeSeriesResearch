import torch

# Need to make sure dataset is sorted by data lengthes from long to short.
class SlidingWindowStepDataProcessor:
    def __init__(self, preProcessor, postProcessor, windowSize, step):
        self.preProcessor = preProcessor
        self.postProcessor = postProcessor
        self.windowSize = windowSize
        self.step = step

    def process(self, data, lengths):
        shuffleIdx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffleIdx, :, :]
        lengths = lengths[shuffleIdx]
        return dataset, lengths