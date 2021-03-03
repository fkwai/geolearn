import time
import os
import numpy as np
import pandas as pd
import torch
import json
from datetime import date
import warnings
from hydroDL import kPath, utils
from hydroDL.data import usgs, gageII, gridMET, transform, dbBasin
from hydroDL.model import rnn, crit, trainBasin
# from sklearn.linear_model import LinearRegression


defaultMaster = dict(
    dataName='test', trainName='all', outName=None,
    hiddenSize=256, batchSize=[365, 500],
    nEpoch=500, saveEpoch=100, resumeEpoch=0,
    optNaN=[1, 1, 0, 0], overwrite=True,
    modelName='CudnnLSTM', crit='RmseLoss', optim='AdaDelta',
    varX=gridMET.varLst, varXC=gageII.lstWaterQuality,
    varY=['00060'], varYC=None,
    sd='1979-01-01', ed='2010-01-01', subset='all', borrowStat=None
)


def nameFolder(outName):
    outFolder = os.path.join(kPath.dirWQ, 'modelFull', outName)
    return outFolder


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
    return dictPar


def loadMaster(outName):
    modelFolder = nameFolder(outName)
    masterFile = os.path.join(modelFolder, 'master.json')
    with open(masterFile, 'r') as fp:
        master = json.load(fp)
    mm = defaultMaster.copy()
    mm.update(master)
    return mm


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
        optFile = os.path.join(outFolder, 'optim_ep{}'.format(ep))
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


def trainModel(outName):
    outFolder = nameFolder(outName)
    dictP = loadMaster(outName)

    # load data
    DM = dbBasin.DataModelFull(dictP['dataName'])
    varTup = (dictP['varX'], dictP['varXC'], dictP['varY'], dictP['varYC'])
    dataTup = DM.extractData(
        varTup, dictP['subset'], dictP['sd'], dictP['ed'])
    if dictP['borrowStat'] is None:
        dataTup, statTup = DM.transIn(dataTup, varTup)
    else:
        statTup = loadStat(dictP['borrowStat'])
        dataTup = DM.transIn(dataTup, varTup, statTup=statTup)
    dataTup = trainBasin.dealNaN(dataTup, dictP['optNaN'])
    wrapStat(outName, statTup)

    # train model
    [nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup)
    # define loss
    lossFun = getattr(crit, dictP['crit'])()
    if dictP['crit'] == 'SigmaLoss':
        ny = ny*2
        nyc = nyc*2
    # define model
    if dictP['modelName'] == 'CudnnLSTM':
        model = rnn.CudnnLstmModel(
            nx=nx+nxc, ny=ny+nyc, hiddenSize=dictP['hiddenSize'])
    elif dictP['modelName'] == 'LstmModel':
        model = rnn.LstmModel(
            nx=nx+nxc, ny=ny+nyc, hiddenSize=dictP['hiddenSize'])
    else:
        raise RuntimeError('Model not specified')

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    if dictP['optim'] == 'AdaDelta':
        optim = torch.optim.Adadelta(model.parameters())
    else:
        raise RuntimeError('optimizor function not specified')

    lossLst = list()
    nEp = dictP['nEpoch']
    sEp = dictP['saveEpoch']
    logFile = os.path.join(outFolder, 'log')
    if os.path.exists(logFile):
        os.remove(logFile)
    for k in range(0, nEp, sEp):
        model, optim, lossEp = trainBasin.trainModel(
            dataTup, model, lossFun, optim, batchSize=dictP['batchSize'],
            nEp=sEp, cEp=k, logFile=logFile)
        # save model
        saveModel(outName, k+sEp, model, optim=optim)
        lossLst = lossLst+lossEp

    lossFile = os.path.join(outFolder, 'loss.csv')
    pd.DataFrame(lossLst).to_csv(lossFile, index=False, header=False)


def testModel(outName,  DM=None, testSet='all', ep=None, reTest=False, batchSize=20):
    # load master
    master = loadMaster(outName)
    if ep is None:
        ep = master['nEpoch']
    outFolder = nameFolder(outName)
    testFileName = 'testP-{}-Ep{}.npz'.format(testSet, ep)
    testFile = os.path.join(outFolder, testFileName)

    if os.path.exists(testFile) and reTest is False:
        print('load saved test result')
        npz = np.load(testFile, allow_pickle=True)
        yP = npz['yP']
        ycP = npz['ycP']
    else:
        statTup = loadStat(outName)
        model = loadModel(outName, ep=ep)
        # load test data
        if DM is None:
            DM = dbBasin.DataModelFull(master['dataName'])
        varTup = (master['varX'], master['varXC'],
                  master['varY'], master['varYC'])
        # test for full sequence for now
        sd = '1979-01-01'
        ed = '2020-01-01'
        dataTup = DM.extractData(varTup, testSet, sd, ed)
        dataTup = DM.transIn(dataTup, varTup, statTup=statTup)
        sizeLst = trainBasin.getSize(dataTup)
        if master['optNaN'] == [2, 2, 0, 0]:
            master['optNaN'] = [0, 0, 0, 0]
        dataTup = trainBasin.dealNaN(dataTup, master['optNaN'])
        x = dataTup[0]
        xc = dataTup[1]
        ny = sizeLst[2]
        # test model - point by point
        yOut, ycOut = trainBasin.testModel(
            model, x, xc, ny, batchSize=batchSize)
        yP = DM.transOut(yOut, statTup[2], master['varY'])
        ycP = DM.transOut(ycOut, statTup[3], master['varYC'])
        np.savez(testFile, yP=yP, ycP=ycP)
    return yP, ycP


def getObs(outName, testSet, DM=None):
    master = loadMaster(outName)
    sd = '1979-01-01'
    ed = '2020-01-01'
    if DM is None:
        DM = dbBasin.DataModelFull(master['dataName'])
    varTup = (master['varX'], master['varXC'], master['varY'], master['varYC'])
    dataTup = DM.extractData(varTup, testSet, sd, ed)
    yT, ycT = dataTup[2:]
    return yT, ycT
