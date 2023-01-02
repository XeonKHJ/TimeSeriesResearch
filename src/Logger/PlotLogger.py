from Logger.ILogger import ILogger
import matplotlib.pyplot

class PlotLogger(ILogger):
    def __init__(self):
        pass

    def logInfo(self, folder):
        pass

    def logResult(self, ogData, predictData):
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(ogData, label="dataset")
        ax.plot(predictData, label="predict")
        ax.legend()
        matplotlib.pyplot.show()