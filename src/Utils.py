import torch

class DynamicThreshold(object):
    def __init__(self, rate, windowSize) -> None:
        self.rate = rate
        self.windowSize = windowSize

    def getThreshold(self, tensor):
        threasholds = torch.zeros(tensor.shape)
        for idx in range(0, self.windowSize, self.windowSize):
            curWindowData = tensor[:, idx:idx+self.windowSize, :]
            std, mean = torch.std_mean(curWindowData)
            threshold = std + self.rate * mean
            threasholds[:, idx:idx+self.windowSize, :] = threshold = std + self.rate * mean
        return threasholds

    def compare(self, thresholds, originalTensor, evalTensor):
        diffs = torch.zeros(originalTensor.shape)
        for idx in range(0, self.windowSize, self.windowSize):
            windowedOgData = originalTensor[:, idx:idx+self.windowSize, :]
            windowedEvalData = evalTensor[:, idx:idx+self.windowSize, :]
            diff = torch.sqrt(torch.square(windowedOgData - windowedEvalData).sum()/self.windowSize)
            diffs[:, idx:idx+self.windowSize, :] = diff
        return diffs > thresholds