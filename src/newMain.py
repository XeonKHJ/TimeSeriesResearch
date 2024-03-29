from Experiment.AheadWithErrorGruAENABArtiDailyExperiment import AheadWithErrorGruAENABArtiDailyExperiment
from Experiment.AheadWithErrorGruAENATAwsEC2CPUExperiment import AheadWithErrorGruAENATAwsEC2CPUExperiment
import torch
import torch.nn
import os.path as path

from Dataset.RegularDataset import RegularDataset
from Experiment.AheadGruAEArtiExperiment import AheadGruAEArtiExperiment
from Experiment.AheadWithErrorGruAEArtiExperiment import AheadWithErrorGruAEArtiExperiment
from Experiment.AheadWithErrorGruAENATAwsEC2DiskWriteExperiment import AheadWithErrorGruAENATAwsEC2DiskWriteExperiment
from Experiment.AheadWithErrorGruAENATAwsEC2NetworkExperiment import AheadWithErrorGruAENATAwsEC2NetworkExperiment
from Experiment.AheadWithErrorGruAENATAwsElbReqExperiment import AheadWithErrorGruAENATAwsElbReqExperiment
from Experiment.AheadWithErrorGruAENATExCpcExperiment import AheadWithErrorGruAENATExCpcExperiment
from Experiment.AheadWithErrorGruAENATExCpmExperiment import AheadWithErrorGruAENATExCpmExperiment
from Experiment.AheadWithErrorGruAENATTraOccuExperiment import AheadWithErrorGruAENATTraOccuExperiment
from Experiment.AheadWithErrorGruAENATTraSpdExperiment import AheadWithErrorGruAENATTraSpdExperiment
from Experiment.AheadWithErrorGruAENABTweetExperiment import AheadWithErrorGruAENABTweetExperiment
from Experiment.AheadWithErrorGruAENATTraTtExperiment import AheadWithErrorGruAENATTraTtExperiment
from Experiment.GeneratedRAENABArtiExperiment import GeneratedRAENABArtiExperiment
from Experiment.GruAENABAWSExperiment import GruAENABAWSExperiment
from Experiment.OneDGruAENABArtDailyExperiment import OneDGruAENABArtDailyExperiment
from Experiment.OneDGruAENABAwsEC2CPUExperiment import OneDGruAENABAwsEC2CPUExperiment
from Experiment.OneDGruAENABAwsEC2DiskWriteExperiment import OneDGruAENABAwsEC2DiskWriteExperiment
from Experiment.OneDGruAENABAwsEC2NetwworkInExperiment import OneDGruAENABAwsEC2NetworkInExperiment
from Experiment.OneDGruAENABAwsElbReqExperiment import OneDGruAENABAwsElbReqExperiment
from Experiment.OneDGruAENABExCpcExperiment import OneDGruAENABExCpcExperiment
from Experiment.OneDGruAENABExCpmExperiment import OneDGruAENABExCpmExperiment
from Experiment.OneDGruAENABTraOccuExperiment import OneDGruAENABTraOccuExperiment
from Experiment.OneDGruAENABTraSpdExperiment import OneDGruAENABTraSpdExperiment
from Experiment.OneDGruAENABTraTtExperiment import OneDGruAENABTraTtExperiment
from Experiment.OneDGruAENABTweetExperiment import OneDGruAENABTweetExperiment
from Experiment.RAENABExperiment import RAENABExperiment
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

    # experiment = RAENABExperiment(logger, "realAWSCloudwatch", "network_in")
    experiment = AheadWithErrorGruAENABTweetExperiment(logger)
    trainer, trainDataReader, validDataReader, processers, datasetSeperator, dataNormalizer = experiment.getExperimentConfig()

    # load data
    fullDataTensor, fullDataLenghts, fullDataLabels, fileList = trainDataReader.read()
    validDataTensor, validDataLengths, validDataLabels, validFileList = validDataReader.read()

    dataNormalizer.addDatasetToRef(fullDataTensor, fullDataLenghts)
    fullDataTensor = dataNormalizer.normalizeDataset(fullDataTensor, fullDataLenghts)
    fullDataTensor = torch.cat((fullDataTensor, fullDataLabels), 2)
    validDataTensor = dataNormalizer.normalizeDataset(validDataTensor, validDataLengths)
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

    trainDataLoader = DataLoader(trainDataset, batch_size=1000, shuffle=False)
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
                trainer.evalResult(validData, lengths, labels)  
                trainer.recordResult(validData, lengths, newFileList)           
        epoch += 1