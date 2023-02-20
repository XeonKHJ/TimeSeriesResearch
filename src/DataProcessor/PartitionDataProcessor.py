import torch

# Need to make sure dataset is sorted by data lengthes from long to short.
class PartitionDataProcessor:
    def __init__(self, partition):
        self.partition = partition

    def process(self, dataset, lengths):
        lengths = (lengths * self.partition).int()
        maxLength = lengths.max().item()
        dataset = dataset[:, 0:maxLength, :]
        return dataset, lengths