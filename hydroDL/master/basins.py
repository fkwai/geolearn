import time
import os
import numpy as np
import pandas as pd
import torch
import json
from datetime import date
import warnings
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, transform
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit, trainTS


def wrapMaster(dataName, trainName, saveName=None, modelName='CudnnLSTM',
               hiddenSize=256, batchSize=[365, 100], nEpoch=500, saveEpoch=100, resumeEpoch=0,
               optNaN=[1, 1, 0, 0], optQ=1, overwrite=True):
    # default parameters
    dictPar = dict(dataName=dataName, trainName=trainName, saveName=saveName, modelName=modelName,
                   hiddenSize=hiddenSize, batchSize=batchSize,
                   nEpoch=nEpoch, saveEpoch=saveEpoch, resumeEpoch=resumeEpoch,
                   optNaN=optNaN, optQ=optQ)

    # create model folder
    if saveName is None:
        saveName = dataName+'_'+trainName
    saveFolder = os.path.join(kPath.dirWQ, 'model', saveName)
    if os.path.exists(saveFolder):
        if overwrite is False:
            saveName = saveFolder+'_'+date.today().strftime("%Y%m%d")
            saveFolder = os.path.join(kPath.dirWQ, 'model', saveName)
            if os.path.exists(saveFolder):
                print('overwrite in folder: '+saveName)
            else:
                os.mkdir(saveFolder)
    else:
        os.mkdir(saveFolder)
    dictPar['saveName'] = saveName
    with open(os.path.join(saveFolder, 'master.json'), 'w') as fp:
        json.dump(dictPar, fp)
    return saveName


def loadMaster(saveName):
    modelFolder = os.path.join(kPath.dirWQ, 'model', saveName)
    masterFile = os.path.join(modelFolder, 'master.json')
    with open(masterFile, 'r') as fp:
        master = json.load(fp)
    return master


def trainModelTS(saveName):
    saveFolder = os.path.join(kPath.dirWQ, 'model', saveName)
    dictP = loadMaster(saveName)

    # load data
    wqData = waterQuality.DataModelWQ(dictP['dataName'])
    dataTup, statTup = wqData.transIn(
        subset=dictP['trainName'], optQ=dictP['optQ'])
    dataTup = trainTS.dealNaN(dataTup, dictP['optNaN'])

    dictStat = dict(statX=statTup[0], statXC=statTup[1],
                    statY=statTup[2], statYC=statTup[3])
    with open(os.path.join(saveFolder, 'stat.json'), 'w') as fp:
        json.dump(dictStat, fp)

    # train model
    [nx, nxc, ny, nyc, nt, ns] = trainTS.getSize(dataTup)
    if dictP['modelName'] == 'CudnnLSTM':
        model = rnn.CudnnLstmModel(
            nx=nx+nxc, ny=ny+nyc, hiddenSize=dictP['hiddenSize'])
    lossFun = crit.RmseLoss()
    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()
    optim = torch.optim.Adadelta(model.parameters())
    lossLst = list()
    for k in range(0, dictP['nEpoch'], dictP['saveEpoch']):
        model, optim, lossEp = trainTS.trainModel(dataTup, model, lossFun, optim, batchSize=dictP['batchSize'],
                                                  nEp=dictP['saveEpoch'], cEp=k)
        # save model
        modelFile = os.path.join(
            saveFolder, 'model_ep{}'.format(k+dictP['saveEpoch']))
        torch.save(model, modelFile)
        modelFile = os.path.join(
            saveFolder, 'optim_ep{}'.format(k+dictP['saveEpoch']))
        torch.save(model, modelFile)
        lossLst = lossLst+lossEp

    lossFile = os.path.join(saveFolder, 'loss.csv')
    pd.DataFrame(lossLst).to_csv(lossFile, index=False, header=False)
