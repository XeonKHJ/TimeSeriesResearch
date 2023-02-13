import torch
import torch.nn
import os.path as path

from Dataset.RegularDataset import RegularDataset
from Experiment.AheadGruAEArtiExperiment import AheadGruAEArtiExperiment
from Experiment.AheadWithErrorGruAEArtiExperiment import AheadWithErrorGruAEArtiExperiment
from Experiment.GeneratedRAENABArtiExperiment import GeneratedRAENABArtiExperiment
from Experiment.OneDGruAENABAwsExperiment import OneDGruAENABAwsExperiment
from Experiment.RAENABArtiExperiment import RAENABArtiExperiment
from Logger.PlotLogger import PlotLogger

from Experiment.GruAENABArtiExperiment import GruAENABArtiExperiment
from Experiment.OneDGruAENABArtiExperiment import OneDGruAENABArtiExperiment
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

    experiment = AheadWithErrorGruAEArtiExperiment(logger)
    trainer, trainDataReader, validDataReader, processers, datasetSeperator, dataNormalizer = experiment.getExperimentConfig()

    # load data
    fullDataTensor, fullDataLenghts, fullDataLabels, fileList = trainDataReader.read()
    validDataTensor, validDataLengths, validDataLabels, validFileList = validDataReader.read()

    dataNormalizer.addDatasetToRef(fullDataTensor)
    fullDataTensor = dataNormalizer.normalizeDataset(fullDataTensor)
    fullDataTensor = torch.cat((fullDataTensor, fullDataLabels), 2)
    validDataTensor = dataNormalizer.normalizeDataset(validDataTensor)
    validDataTensor = torch.cat((validDataTensor, validDataLabels), 2)

    # displayDataTensor, displayDataLenghts, displayDataLabels, displayFileList = dataReader.read()
    for i in range(validDataTensor.shape[0]):
        curList = validDataTensor[i,:,0:validDataTensor.shape[2]-1].reshape(-1).tolist()
        logger.logResults([curList], ['real'], path.splitext(path.basename(validFileList[i]))[0], globalConfig.getOriginalPicturePath())

    # data preprocess
    dataTensor = fullDataTensor
    dataLengths = fullDataLenghts
    for processor in processers:
        dataTensor, dataLengths = processor.process(dataTensor, dataLengths)
    trainDataTensor, trainsetLengths = datasetSeperator.getTrainningSet(dataTensor, dataLengths)
    testDataTensor, testSetLengths = datasetSeperator.getValidationSet(dataTensor, dataLengths)

    trainDataset = RegularDataset(trainDataTensor, trainsetLengths)
    testDataset = RegularDataset(testDataTensor, testSetLengths)
    validDataset = RegularDataset(validDataTensor, validDataLengths)

    # start trainning
    epoch = 0
    keepTrainning = True

    trainDataLoader = DataLoader(trainDataset, batch_size=1000, shuffle=True)
    testDataLoader = DataLoader(testDataset, shuffle=False, batch_size=testDataTensor.shape[0])
    validDataLaoder = DataLoader(validDataset, shuffle=False, batch_size = validDataTensor.shape[0])

    while keepTrainning:
        for trainData, trainLabels in trainDataLoader:
            lengths = trainLabels[:, trainLabels.shape[1]-1]
            labels = trainLabels[:, 0:trainLabels.shape[1]-1]
            loss = trainer.train(trainData, lengths, labels)
        if epoch % 50 == 0:
            trainer.save()
            # for testData, testLabels in testDataLoader:
            #     lengths = testLabels[:, testLabels.shape[1]-1]
            #     labels = testLabels[:, 0:testLabels.shape[1]-1]            
            #     trainer.evalResult(testData, lengths, labels)
            for validData, validLabels in validDataLaoder:
                lengths = validLabels[:, validLabels.shape[1]-1]
                labels = validLabels[:, 0:validLabels.shape[1]-1] 
                newFileList = list() 
                for fileName in validFileList:
                     newFileList.append(path.splitext(path.basename(fileName))[0])
                trainer.recordResult(validData, lengths, newFileList)
                trainer.evalResult(validData, lengths, labels)                  
        epoch += 1