import torch
import torch.nn
import os.path as path

from Dataset.RegularDataset import RegularDataset
from Logger.PlotLogger import PlotLogger

from Experiment.GruAENABArtiExperiment import GruAENABArtiExperiment
from globalConfig import globalConfig

import ArgParser
from torch.utils.data import DataLoader

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is avaliable.")
    else:
        print("CUDA is unavaliable")

    # read arg
    args = ArgParser.getArgs()

    logger = PlotLogger((not args.disablePlot))

    experiment = GruAENABArtiExperiment(logger)
    trainer, dataReader, processers, datasetSeperator, dataNormalizer = experiment.getExperimentConfig()

    # load data
    fullDataTensor, fullDataLenghts, fullDataLabels, fileList = dataReader.read()

    dataNormalizer.addDatasetToRef(fullDataTensor)
    fullDataTensor = dataNormalizer.normalizeDataset(fullDataTensor)
    fullDataTensor = torch.cat((fullDataTensor, fullDataLabels), 2)

    # displayDataTensor, displayDataLenghts, displayDataLabels, displayFileList = dataReader.read()
    for i in range(fullDataTensor.shape[0]):
        curList = fullDataTensor[i,:,0:fullDataTensor.shape[2]-1].reshape(-1).tolist()
        logger.logResults([curList], ['real'], path.splitext(path.basename(fileList[i]))[0], globalConfig.getOriginalPicturePath())

    # data preprocess
    dataTensor = fullDataTensor
    dataLengths = fullDataLenghts
    for processor in processers:
        dataTensor, dataLengths = processor.process(dataTensor, dataLengths)
    trainDataTensor, trainsetLengths = datasetSeperator.getTrainningSet(dataTensor, dataLengths)
    testDataTensor, testSetLengths = datasetSeperator.getValidationSet(dataTensor, dataLengths)

    trainDataset = RegularDataset(trainDataTensor, trainsetLengths)
    testDataset = RegularDataset(testDataTensor, testSetLengths)
    validDataset = RegularDataset(fullDataTensor, fullDataLenghts)

    # start trainning
    epoch = 0
    keepTrainning = True

    trainDataLoader = DataLoader(trainDataset, batch_size=1000, shuffle=True)
    testDataLoader = DataLoader(testDataset, shuffle=False, batch_size=testDataTensor.shape[0])
    validDataLaoder = DataLoader(validDataset, shuffle=False, batch_size = fullDataTensor.shape[0])

    while keepTrainning:
        for trainData, trainLabels in trainDataLoader:
            lengths = trainLabels[:, trainLabels.shape[1]-1]
            labels = trainLabels[:, 0:trainLabels.shape[1]-1]
            loss = trainer.train(trainData, lengths, labels)
        if epoch % 100 == 0:
            trainer.save()
            # for testData, testLabels in testDataLoader:
            #     lengths = testLabels[:, testLabels.shape[1]-1]
            #     labels = testLabels[:, 0:testLabels.shape[1]-1]            
            #     trainer.evalResult(testData, lengths, labels)
            for validData, validLabels in validDataLaoder:
                lengths = validLabels[:, validLabels.shape[1]-1]
                labels = validLabels[:, 0:validLabels.shape[1]-1] 
                newFileList = list() 
                for fileName in fileList:
                     newFileList.append(path.splitext(path.basename(fileName))[0])
                trainer.recordResult(validData, lengths, newFileList)
                trainer.evalResult(validData, lengths, labels)                  
        epoch += 1