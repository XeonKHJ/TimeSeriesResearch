import torch

class DynamicThreshold(object):
    def __init__(self, meanRate=0.3, stdRate=0.5, windowSize=100) -> None:
        self.rate = meanRate
        self.windowSize = windowSize

    def getThreshold(self, tensor):
        threasholds = torch.zeros(tensor.shape)
        for idx in range(0, tensor.shape[1], self.windowSize):
            curWindowData = tensor[:, idx:idx+self.windowSize, :]
            std, mean = torch.std_mean(curWindowData, 1)
            threshold = std + self.rate * mean
            threasholds[:, idx:idx+self.windowSize, :] = threshold
        return threasholds

    def compare(self, thresholds, originalTensor, evalTensor):
        diffs = torch.zeros(originalTensor.shape)
        for idx in range(0, originalTensor.shape[1], self.windowSize):
            windowedOgData = originalTensor[:, idx:idx+self.windowSize, :]
            windowedEvalData = evalTensor[:, idx:idx+self.windowSize, :]
            diff = torch.sqrt(torch.square(windowedOgData - windowedEvalData).sum(1)/self.windowSize)
            diffs[:, idx:idx+self.windowSize, :] = diff
        return diffs > thresholds

class PaperDynamicThreshold(object):
    def __init__(self, rate, windowSize) -> None:
        self.rate = rate
        self.windowSize = windowSize

    def getThreshold(self, tensor):


        threasholds = torch.zeros(tensor.shape)
        for idx in range(0, tensor.shape[1], self.windowSize):
            curWindowData = tensor[:, idx:idx+self.windowSize, :]
            std, mean = torch.std_mean(curWindowData, 1)
            threshold = std + self.rate * mean
            threasholds[:, idx:idx+self.windowSize, :] = threshold
        return threasholds

    def compare(self, thresholds, originalTensor, evalTensor):
        diffs = torch.zeros(originalTensor.shape)
        for idx in range(0, originalTensor.shape[1], self.windowSize):
            windowedOgData = originalTensor[:, idx:idx+self.windowSize, :]
            windowedEvalData = evalTensor[:, idx:idx+self.windowSize, :]
            diff = torch.sqrt(torch.square(windowedOgData - windowedEvalData).sum(1)/self.windowSize)
            diffs[:, idx:idx+self.windowSize, :] = diff
        return diffs > thresholds