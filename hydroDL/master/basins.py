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


def nameFolder(outName):
    outFolder = os.path.join(kPath.dirWQ, 'model', outName)
    return outFolder


defaultMaster = dict(
    dataName='HBN', trainName='first50', outName=None, modelName='CudnnLSTM',
    hiddenSize=256, batchSize=[None, 500], nEpoch=500, saveEpoch=100, resumeEpoch=0,
    optNaN=[1, 1, 0, 0], overwrite=True,
    varX=gridMET.varLst, varXC=gageII.lstWaterQuality,
    varY=usgs.varQ, varYC=usgs.varC
)


def wrapMaster(**kw):
    # default parameters
    dictPar = defaultMaster.copy()
    dictPar.update(kw)
    diff = list(set(dictPar) - set(defaultMaster))
    if len(diff) > 0:
        raise Exception('parameters not understand: '+' '.join(diff))

    # create model folder
    if dictPar['outName'] is None:
        dictPar['outName'] = dictPar['dataName']+'_'+dictPar['trainName']
    outFolder = nameFolder(dictPar['outName'])
    if os.path.exists(outFolder):
        if dictPar['overwrite'] is False:
            outName = outFolder+'_'+date.today().strftime("%Y%m%d")
            outFolder = os.path.join(kPath.dirWQ, 'model', outName)
            if os.path.exists(outFolder):
                print('overwrite in folder: '+outName)
            else:
                os.mkdir(outFolder)
            dictPar['outName'] = outName
    else:
        os.mkdir(outFolder)
    with open(os.path.join(outFolder, 'master.json'), 'w') as fp:
        json.dump(dictPar, fp)
    return dictPar['outName']


def loadMaster(outName):
    modelFolder = os.path.join(kPath.dirWQ, 'model', outName)
    masterFile = os.path.join(modelFolder, 'master.json')
    with open(masterFile, 'r') as fp:
        master = json.load(fp)
    return master


def wrapStat(outName, statTup):
    outFolder = nameFolder(outName)
    dictStat = dict(statX=statTup[0], statXC=statTup[1],
                    statY=statTup[2], statYC=statTup[3])
    with open(os.path.join(outFolder, 'stat.json'), 'w') as fp:
        json.dump(dictStat, fp)


def loadStat(outName):
    outFolder = nameFolder(outName)
    statFile = os.path.join(outFolder, 'stat.json')
    with open(statFile, 'r') as fp:
        dictStat = json.load(fp)
        statX = dictStat['statX']
        statXC = dictStat['statXC']
        statY = dictStat['statY']
        statYC = dictStat['statYC']
    return statX, statXC, statY, statYC


def loadModel(outName, ep=None, opt=False):
    outFolder = nameFolder(outName)
    if ep is None:
        mDict = loadMaster(outName)
        ep = mDict['nEpoch']
    modelFile = os.path.join(outFolder, 'model_ep{}'.format(ep))
    model = torch.load(modelFile)
    if opt:
        optFile = os.path.join(outFolder, 'optim_ep{}'.format(k+sEp))
        optim = torch.load(optFile)
        return model, optim
    else:
        return model


def saveModel(outName, ep, model, optim=None):
    outFolder = nameFolder(outName)
    modelFile = os.path.join(outFolder, 'model_ep{}'.format(ep))
    torch.save(model, modelFile)
    if optim is not None:
        optFile = os.path.join(outFolder, 'optim_ep{}'.format(ep))
        torch.save(optim, optFile)


def trainModelTS(outName):
    outFolder = nameFolder(outName)
    dictP = loadMaster(outName)

    # load data
    wqData = waterQuality.DataModelWQ(dictP['dataName'])
    varTup = (dictP['varX'], dictP['varXC'], dictP['varY'], dictP['varYC'])
    dataTup, statTup = wqData.transIn(
        subset=dictP['trainName'], varTup=varTup)
    dataTup = trainTS.dealNaN(dataTup, dictP['optNaN'])
    wrapStat(outName, statTup)

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
    nEp = dictP['nEpoch']
    sEp = dictP['saveEpoch']
    logFile = os.path.join(outFolder, 'log')
    if os.path.exists(logFile):
        os.remove(logFile)
    for k in range(0, nEp, sEp):
        model, optim, lossEp = trainTS.trainModel(
            dataTup, model, lossFun, optim, batchSize=dictP['batchSize'],
            nEp=sEp, cEp=k, logFile=logFile)
        # save model
        saveModel(outName, k+sEp, model, optim=optim)
        lossLst = lossLst+lossEp

    lossFile = os.path.join(outFolder, 'loss.csv')
    pd.DataFrame(lossLst).to_csv(lossFile, index=False, header=False)


def testModel(outName, testset, wqData=None, ep=None):
    # load master
    master = loadMaster(outName)
    statTup = loadStat(outName)
    model = loadModel(outName, ep=ep)

    # load test data
    wqData = waterQuality.DataModelWQ(master['dataName'])
    varTup = (master['varX'], master['varXC'], master['varY'], master['varYC'])

    testDataLst = wqData.transIn(
        subset=testset, statTup=statTup, varTup=varTup)
    sizeLst = trainTS.getSize(testDataLst)
    testDataLst = trainTS.dealNaN(testDataLst, master['optNaN'])
    x = testDataLst[0]
    xc = testDataLst[1]
    ny = sizeLst[2]

    # test model - point by point
    yOut, ycOut = trainTS.testModel(model, x, xc, ny)
    qP = wqData.transOut(yOut, statTup[2], master['varY'])
    cP = wqData.transOut(ycOut, statTup[3], master['varYC'])
    obsLst = wqData.extractSubset(testset)
    qT, cT = obsLst[2:]
    return cP, cT
