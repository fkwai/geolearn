import time
import os
import numpy as np
import pandas as pd
import torch
import json
from datetime import date
import warnings
from hydroDL import kPath, utils
from hydroDL.data import usgs, gageII, gridMET, transform
# from hydroDL.app import waterQuality
from hydroDL.app import waterQuality2 as waterQuality
from hydroDL.model import rnn, crit, trainTS
# from sklearn.linear_model import LinearRegression


defaultMaster = dict(
    dataName='HBN', trainName='first50', outName=None, modelName='CudnnLSTM',
    hiddenSize=256, batchSize=[None, 500], nEpoch=500, saveEpoch=100, resumeEpoch=0,
    optNaN=[1, 1, 0, 0], overwrite=True,
    varX=gridMET.varLst, varXC=gageII.lstWaterQuality,
    varY=usgs.varQ, varYC=usgs.varC
)


def nameFolder(outName):
    outFolder = os.path.join(kPath.dirWQ, 'model', outName)
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


def getObs(outName, testset, wqData=None):
    master = loadMaster(outName)
    varTup = (master['varX'], master['varXC'], master['varY'], master['varYC'])
    if wqData is None:
        wqData = waterQuality.DataModelWQ(master['dataName'])
    dataTup = wqData.extractData(varTup=varTup, subset=testset)
    yT, ycT = dataTup[2:]
    return yT, ycT


def testModel(outName, testset, wqData=None, ep=None, reTest=False):
    # load master
    master = loadMaster(outName)
    if ep is None:
        ep = master['nEpoch']
    outFolder = nameFolder(outName)
    testFileName = 'testP-{}-Ep{}.npz'.format(testset, ep)
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
        if wqData is None:
            wqData = waterQuality.DataModelWQ(master['dataName'])
        varTup = (master['varX'], master['varXC'],
                  master['varY'], master['varYC'])
        testDataLst = wqData.transIn(
            subset=testset, statTup=statTup, varTup=varTup)
        sizeLst = trainTS.getSize(testDataLst)
        testDataLst = trainTS.dealNaN(testDataLst, master['optNaN'])
        x = testDataLst[0]
        xc = testDataLst[1]
        ny = sizeLst[2]
        # test model - point by point
        yOut, ycOut = trainTS.testModel(model, x, xc, ny)
        yP = wqData.transOut(yOut, statTup[2], master['varY'])
        ycP = wqData.transOut(ycOut, statTup[3], master['varYC'])
        np.savez(testFile, yP=yP, ycP=ycP)
    return yP, ycP


def testModelSeq(outName, siteNoLst, wqData=None, ep=None, returnOut=False,
                 sd=np.datetime64('1979-01-01'),
                 ed=np.datetime64('2020-01-01')):
    # run sequence test for all sites, default to be from first date to last date
    if type(siteNoLst) is not list:
        siteNoLst = [siteNoLst]
    master = loadMaster(outName)
    if ep is None:
        ep = master['nEpoch']
    outDir = nameFolder(outName)
    sdS = pd.to_datetime(sd).strftime('%Y%m%d')
    edS = pd.to_datetime(ed).strftime('%Y%m%d')
    saveDir = os.path.join(outDir, 'seq-{}-{}-ep{}'.format(sdS, edS, ep))
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    siteSaveLst = os.listdir(saveDir)
    sitePredLst = [siteNo for siteNo in siteNoLst if siteNo not in siteSaveLst]
    if len(sitePredLst) != 0:
        if wqData is None:
            wqData = waterQuality.DataModelWQ(master['dataName'])
        (varX, varXC, varY, varYC) = (
            master['varX'], master['varXC'], master['varY'], master['varYC'])
        (statX, statXC, statY, statYC) = loadStat(outName)
        model = loadModel(outName, ep=ep)
        tabG = gageII.readData(varLst=varXC, siteNoLst=siteNoLst)
        tabG = gageII.updateCode(tabG)
        for siteNo in sitePredLst:
            if 'DRAIN_SQKM' in varXC:
                area = tabG[tabG.index == siteNo]['DRAIN_SQKM'].values[0]
            else:
                area = None
            # test model
            print('testing {} from {} to {}'.format(siteNo, sdS, edS))
            dfX = waterQuality.readSiteX(
                siteNo, varX, sd=sd, ed=ed, area=area, nFill=5)
            xA = np.expand_dims(dfX.values, axis=1)
            xcA = np.expand_dims(
                tabG.loc[siteNo].values.astype(np.float), axis=0)
            mtdX = wqData.extractVarMtd(varX)
            x = transform.transInAll(xA, mtdX, statLst=statX)
            mtdXC = wqData.extractVarMtd(varXC)
            xc = transform.transInAll(xcA, mtdXC, statLst=statXC)
            yOut = trainTS.testModel(model, x, xc)
            # transfer out
            nt = len(dfX)
            ny = len(varY) if varY is not None else 0
            nyc = len(varYC) if varYC is not None else 0
            yP = np.full([nt, ny+nyc], np.nan)
            yP[:, :ny] = wqData.transOut(yOut[:, 0, :ny], statY, varY)
            yP[:, ny:] = wqData.transOut(yOut[:, 0, ny:], statYC, varYC)
            # save output
            t = dfX.index.values.astype('datetime64[D]')
            colY = [] if varY is None else varY
            colYC = [] if varYC is None else varYC
            dfOut = pd.DataFrame(data=yP, columns=[colY+colYC], index=t)
            dfOut.index.name = 'date'
            dfOut = dfOut.reset_index()
            dfOut.to_csv(os.path.join(saveDir, siteNo), index=False)
    # load all csv
    if returnOut:
        dictOut = dict()
        for siteNo in siteNoLst:
            # print('loading {} from {} to {}'.format(siteNo, sdS, edS))
            dfOut = pd.read_csv(os.path.join(saveDir, siteNo))
            dictOut[siteNo] = dfOut
        return dictOut


def loadSeq(outName, siteNo,
            sd=np.datetime64('1979-01-01'),
            ed=np.datetime64('2020-01-01'),
            ep=500):
    outDir = nameFolder(outName)
    sdS = pd.to_datetime(sd).strftime('%Y%m%d')
    edS = pd.to_datetime(ed).strftime('%Y%m%d')
    saveDir = os.path.join(outDir, 'seq-{}-{}-ep{}'.format(sdS, edS, ep))
    dfPred = pd.read_csv(os.path.join(saveDir, siteNo))
    dfPred = utils.time.datePdf(dfPred)
    dfObs = waterQuality.readSiteY(siteNo, dfPred.columns.tolist())
    return dfPred, dfObs


def modelLinear(outName, testset, trainset=None, wqData=None):
    master = loadMaster(outName)
    dataName = master['dataName']
    if wqData is None:
        wqData = waterQuality.DataModelWQ(dataName)
    if trainset is None:
        trainset = master['trainName']
    infoTrain = wqData.info.iloc[wqData.subset[trainset]].reset_index()
    infoTest = wqData.info.iloc[wqData.subset[testset]].reset_index()

    # linear reg data
    statTup = loadStat(outName)
    varTup = (master['varX'], master['varXC'], master['varY'], master['varYC'])
    dataTup1 = wqData.transIn(subset=trainset, varTup=varTup, statTup=statTup)
    dataTup2 = wqData.transIn(subset=testset, varTup=varTup, statTup=statTup)
    dataTup1 = trainTS.dealNaN(dataTup1, master['optNaN'])
    dataTup2 = trainTS.dealNaN(dataTup2, master['optNaN'])
    varYC = varTup[3]
    statYC = statTup[3]
    x1 = dataTup1[0][-1, :, :]
    yc1 = dataTup1[3]
    x2 = dataTup2[0][-1, :, :]

    # point test l2 - linear
    nc = len(varYC)
    matP1 = np.full([len(infoTrain), nc], np.nan)
    matP2 = np.full([len(infoTest), nc], np.nan)
    siteNoLst = infoTest['siteNo'].unique().tolist()
    for siteNo in siteNoLst:
        ind1 = infoTrain[infoTrain['siteNo'] == siteNo].index
        ind2 = infoTest[infoTest['siteNo'] == siteNo].index
        xT1 = x1[ind1, :]
        ycT1 = yc1[ind1, :]
        for ic in range(nc):
            [xx, yy], iv = utils.rmNan([xT1, ycT1[:, ic]])
            if len(iv) > 0:
                modelYC = LinearRegression().fit(xx, yy)
                matP1[ind1, ic] = modelYC.predict(xT1)
                if len(ind2) > 0:
                    xT2 = x2[ind2, :]
                    matP1[ind2, ic] = modelYC.predict(xT2)
    matO1 = wqData.transOut(matP1, statYC, varYC)
    matO2 = wqData.transOut(matP2, statYC, varYC)
    return matO1, matO2
