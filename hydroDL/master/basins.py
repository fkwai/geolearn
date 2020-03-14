import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from datetime import date
import warnings


from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, transform
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit, trainTS


def trainModelTS(dataName, trainName, saveName=None, modelName='CudnnLSTM',
                 hiddenSize=256, batchSize=[100, 365], nEpoch=500, saveEpoch=100, resumeEpoch=0,
                 fillNan=[True, True, False, False], rmNan=[False, False, False, False]):
    # default parameters
    dictPar = dict(dataName=dataName, trainName=trainName, saveName=saveName, modelName=modelName,
                   hiddenSize=hiddenSize, batchSize=batchSize,
                   nEpoch=nEpoch, saveEpoch=saveEpoch, resumeEpoch=resumeEpoch,
                   fillNan=fillNan, rmNan=rmNan)
    # create model folder
    if saveName is None:
        saveName = dataName+'_'+trainName
    saveFolder = os.path.join(kPath.dirWQ, 'model', saveName)
    if os.path.exists(saveFolder):
        saveName = saveFolder+'_'+date.today().strftime("%Y%m%d")
        saveFolder = os.path.join(kPath.dirWQ, 'model', saveName)
        if os.path.exists(saveFolder):
            print('overwrite in folder: '+saveName)
        else:
            os.mkdir(saveFolder)
    else:
        os.mkdir(saveFolder)
    dictPar[saveName] = saveName
    with open(os.path.join(saveFolder, 'master.json'), 'w') as fp:
        json.dump(dictPar, fp)

    # load data
    wqData = waterQuality.DataModelWQ(dataName)
    dataTup, statTup = wqData.transIn(subset=trainName)
    # check if any nan
    rmLst = list()
    for k in range(4):
        # inf to nan
        dataTup[k][np.isinf(dataTup[k])] = np.nan
        indNan = np.where(np.isnan(dataTup[k]))
        if len(indNan[0]) > 0:
            if fillNan[k] is True:
                dataTup[k][indNan] = 0
                print('nan found and filled ', k)
            elif rmNan[k] is True:
                if dataTup[k].ndim == 2:
                    rmLst.append(indNan[0])
                if dataTup[k].ndim == 3:
                    rmLst.append(np.unique(np.where(np.isnan(dataTup[k]))[1]))

    dictStat = dict(statX=statTup[0], statXC=statTup[1],
                    statY=statTup[2], statYC=statTup[3])
    with open(os.path.join(saveFolder, 'stat.json'), 'w') as fp:
        json.dump(dictStat, fp)

    # train model
    [nx, nxc, ny, nyc] = [data.shape[-1] for data in dataTup]
    if modelName == 'CudnnLSTM':
        model = rnn.CudnnLstmModel(nx=nx+nxc, ny=ny+nyc, hiddenSize=hiddenSize)
    lossFun = crit.RmseLoss()
    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()
    optim = torch.optim.Adadelta(model.parameters())
    lossLst = list()
    for k in range(0, nEpoch, saveEpoch):
        model, optim, lossEp = trainTS.trainModel(dataTup, model, lossFun, optim, batchSize=batchSize,
                                                  nEp=saveEpoch, cEp=k)
        # save model
        modelFile = os.path.join(saveFolder, 'model_ep{}'.format(k+saveEpoch))
        torch.save(model, modelFile)
        modelFile = os.path.join(saveFolder, 'optim_ep{}'.format(k+saveEpoch))
        torch.save(model, modelFile)
        lossLst = lossLst.append(lossEp)

    lossFile = os.path.join(saveFolder, 'loss.csv')
    df = pd.DataFrame(lossLst).to_csv(saveFile, index=False, header=False)
