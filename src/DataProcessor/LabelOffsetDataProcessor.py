import torch

# Need to make sure dataset is sorted by data lengthes from long to short.
class LabelOffsetDataProcessor:
    def __init__(self, offsetLength):
        self.offsetLength = offsetLength

    def process(self, dataset, lengths):
        dataset[:, 0:dataset.shape[1] - self.offsetLength,:] = dataset[:, self.offsetLength:dataset.shape[1], dataset.shape[1]]
        return dataset, lengths