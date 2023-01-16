from Logger.ILogger import ILogger
import matplotlib.pyplot
import os.path

class PlotLogger(ILogger):
    def __init__(self):
        pass

    def logInfo(self, folder):
        pass

    def logSingleResult(self, data, label):
        _, ax = matplotlib.pyplot.subplots()
        ax.plot(data, label=label)
        ax.legend()
        matplotlib.pyplot.show()

    def logResults(self, datas, labels, picname=None):
        _, ax = matplotlib.pyplot.subplots()
        for i in range(len(datas)):
            ax.plot(datas[i], label = labels[i])
        ax.legend()
        matplotlib.pyplot.show()
        if picname != None:
            matplotlib.pyplot.savefig(os.path.join('SavedPic', picname))
            print(picname, " saved.")

    def logResult(self, ogData, predictData):
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(ogData, label="dataset")
        ax.plot(predictData, label="predict")
        ax.legend()
        matplotlib.pyplot.show()