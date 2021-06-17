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


defaultMaster = dict(
    dataName='test', trainSet='all', outName=None,
    hiddenSize=256, batchSize=[365, 500],
    nEpoch=500, saveEpoch=100, resumeEpoch=0,
    optNaN=[1, 1, 0, 0], overwrite=True,
    modelName='LstmModel', crit='RmseLoss', optim='AdaDelta',
    varX=gridMET.varLst, varXC=gageII.varLst,
    varY=['00060'], varYC=None, borrowStat=None,
    optBatch='Weight', nIterEp=None,
    mtdX=None, mtdY=None, mtdXC=None, mtdYC=None
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
        dictPar['outName'] = dictPar['dataName']+'_'+dictPar['trainSet']
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
    DF = dbBasin.DataFrameBasin(dictP['dataName'])
    dictVar = {k: dictP[k]
               for k in ('varX', 'varXC', 'varY', 'varYC')}
    DM = dbBasin.DataModelBasin(DF, subset=dictP['trainSet'], **dictVar)
    if dictP['borrowStat'] is not None:
        DM.loadStat(dictP['borrowStat'])
    DM.trans(mtdX=dictP['mtdX'], mtdXC=dictP['mtdXC'],
             mtdY=dictP['mtdY'], mtdYC=dictP['mtdYC'])
    DM.saveStat(outFolder)
    dataTup = DM.getData()
    dataTup = trainBasin.dealNaN(dataTup, dictP['optNaN'])

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
            nEp=sEp, cEp=k, logFile=logFile,
            optBatch=dictP['optBatch'], nIterEp=dictP['nIterEp'])
        # save model
        saveModel(outName, k+sEp, model, optim=optim)
        lossLst = lossLst+lossEp

    lossFile = os.path.join(outFolder, 'loss.csv')
    pd.DataFrame(lossLst).to_csv(lossFile, index=False, header=False)


def testModel(outName,  DF=None, testSet='all', ep=None, reTest=False, batchSize=20):
    # load master
    dictP = loadMaster(outName)
    if ep is None:
        ep = dictP['nEpoch']
    outFolder = nameFolder(outName)
    testFileName = 'testP-{}-Ep{}.npz'.format(testSet, ep)
    testFile = os.path.join(outFolder, testFileName)

    if os.path.exists(testFile) and reTest is False:
        print('load saved test result')
        npz = np.load(testFile, allow_pickle=True)
        yP = npz['yP']
        ycP = npz['ycP']
    else:
        model = loadModel(outName, ep=ep)
        # load test data
        if DF is None:
            DF = dbBasin.DataFrameBasin(dictP['dataName'])
        dictVar = {k: dictP[k]
                   for k in ('varX', 'varXC', 'varY', 'varYC')}
        DM = dbBasin.DataModelBasin(DF, subset=testSet, **dictVar)
        DM.loadStat(outFolder)
        dataTup = DM.getData()
        sizeLst = trainBasin.getSize(dataTup)
        dataTup = trainBasin.dealNaN(dataTup, dictP['optNaN'])
        x = dataTup[0]
        xc = dataTup[1]
        ny = sizeLst[2]
        # test model - point by point
        yOut, ycOut = trainBasin.testModel(
            model, x, xc, ny, batchSize=batchSize)
        yP = DM.transOutY(yOut)
        ycP = DM.transOutYC(ycOut)
        np.savez(testFile, yP=yP, ycP=ycP)
    return yP, ycP

