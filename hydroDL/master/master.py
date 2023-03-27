import os
import hydroDL
from collections import OrderedDict
import numpy as np
import json
from hydroDL import utils
import hydroDL.model
import datetime as dt
import pandas as pd


def dataTs2End(dataTup, rho):
    # assuming yc is none
    x, xc, y, yc = dataTup
    xMatLst = [x[k : -rho + 1 + k, ...] for k in range(rho - 1)] + [x[rho - 1 :, ...]]
    xMat = np.stack(xMatLst, axis=0)
    [jTemp, iTemp, k] = np.where(~np.isnan(y))
    j = jTemp - rho + 1
    i = iTemp[j >= 0]
    j = j[j >= 0]
    xE = xMat[:, j, i, :]
    yE = y[j + rho - 1, i]
    xcE = xc[i, :] if xc is not None else None
    # ycE = yc[i, :] if yc is not None else None
    return (xE, xcE, None, yE)


def wrapMaster(out, optData, optModel, optLoss, optTrain):
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain
    )
    return mDict


def readMasterFile(out):
    mFile = os.path.join(out, 'master.json')
    with open(mFile, 'r') as fp:
        mDict = json.load(fp, object_pairs_hook=OrderedDict)
    print('read master file ' + mFile)
    return mDict


def writeMasterFile(mDict):
    out = mDict['out']
    if not os.path.isdir(out):
        os.makedirs(out)
    mFile = os.path.join(out, 'master.json')
    with open(mFile, 'w') as fp:
        json.dump(mDict, fp, indent=4)
    print('write master file ' + mFile)
    return out


def fixRootDB(rootDB):
    if '/' in rootDB:
        nameDB = rootDB.split('/')[-1]
    elif '\\' in rootDB:
        nameDB = rootDB.split('/')[-1]
    return os.path.join(hydroDL.pathSMAP['dirDB'], nameDB)


def loadModel(out, epoch=None):
    if epoch is None:
        mDict = readMasterFile(out)
        epoch = mDict['train']['nEpoch']
    model = hydroDL.model.train.loadModel(out, epoch)
    return model


def namePred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
    mDict = readMasterFile(out)
    target = mDict['data']['target']
    if type(target) is not list:
        target = [target]
    nt = len(target)
    lossName = mDict['loss']['name']
    if epoch is None:
        epoch = mDict['train']['nEpoch']

    fileNameLst = list()
    for k in range(nt):
        testName = '_'.join([subset, str(tRange[0]), str(tRange[1]), 'ep' + str(epoch)])
        fileName = '_'.join([testName, target[k]])
        fileNameLst.append(fileName)
        if lossName == 'hydroDL.model.crit.SigmaLoss':
            fileName = '_'.join([testName, target[k], 'SigmaX'])
            fileNameLst.append(fileName)
    if doMC is not False:
        mcFileNameLst = list()
        for fileName in fileNameLst:
            fileName = '_'.join([testName, target[k], 'SigmaMC' + str(doMC)])
            mcFileNameLst.append(fileName)
        fileNameLst = fileNameLst + mcFileNameLst

    # sum up to file path list
    filePathLst = list()
    for fileName in fileNameLst:
        if suffix is not None:
            fileName = fileName + '_' + suffix
        filePath = os.path.join(out, fileName + '.csv')
        filePathLst.append(filePath)
    return filePathLst


def readPred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
    mDict = readMasterFile(out)
    dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
    for k in range(len(filePathLst)):
        filePath = filePathLst[k]
        dataPred[:, :, k] = pd.read_csv(filePath, dtype=np.float, header=None).values
    isSigmaX = False
    if mDict['loss']['name'] == 'hydroDL.model.crit.SigmaLoss':
        isSigmaX = True
        pred = dataPred[:, :, ::2]
        sigmaX = dataPred[:, :, 1::2]
    else:
        pred = dataPred


def mvobs(data, mvday, rmNan=True):
    obslen = data.shape[1] - mvday + 1  # The length of training daily data
    ngage = data.shape[0]
    mvdata = np.full((ngage, obslen, 1), np.nan)
    for ii in range(obslen):
        tempdata = data[:, ii : ii + mvday, :]
        tempmean = np.nanmean(tempdata, axis=1)
        mvdata[:, ii, 0] = tempmean[:, 0]
    if rmNan is True:
        mvdata[np.where(np.isnan(mvdata))] = 0
    return mvdata


def loadData(optData, readX=True, readY=True):
    if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=optData['rootDB'], subset=optData['subset'], tRange=optData['tRange']
        )
        if readY is True:
            y = df.getDataTs(
                varLst=optData['target'],
                doNorm=optData['doNorm'][1],
                rmNan=optData['rmNan'][1],
            )
        else:
            y = None

        if readX is True:
            x = df.getDataTs(
                varLst=optData['varT'],
                doNorm=optData['doNorm'][0],
                rmNan=optData['rmNan'][0],
            )
            c = df.getDataConst(
                varLst=optData['varC'],
                doNorm=optData['doNorm'][0],
                rmNan=optData['rmNan'][0],
            )
            if optData['daObs'] > 0:
                nday = optData['daObs']
                sd = utils.time.t2dt(optData['tRange'][0]) - dt.timedelta(days=nday)
                ed = utils.time.t2dt(optData['tRange'][1]) - dt.timedelta(days=nday)
                df = hydroDL.data.dbCsv.DataframeCsv(
                    rootDB=optData['rootDB'], subset=optData['subset'], tRange=[sd, ed]
                )
                obs = df.getDataTs(
                    varLst=optData['target'],
                    doNorm=optData['doNorm'][1],
                    rmNan=optData['rmNan'][1],
                )
                x = (x, obs)
        else:
            x = None
            c = None
    elif eval(optData['name']) is hydroDL.data.camels.DataframeCamels:
        df = hydroDL.data.camels.DataframeCamels(
            subset=optData['subset'], tRange=optData['tRange']
        )
        x = df.getDataTs(
            varLst=optData['varT'],
            doNorm=optData['doNorm'][0],
            rmNan=optData['rmNan'][0],
        )
        y = df.getDataObs(doNorm=optData['doNorm'][1], rmNan=optData['rmNan'][1])
        c = df.getDataConst(
            varLst=optData['varC'],
            doNorm=optData['doNorm'][0],
            rmNan=optData['rmNan'][0],
        )
    else:
        raise Exception('unknown database')
    return df, x, y, c


def train(mDict):
    if mDict is str:
        mDict = readMasterFile(mDict)
    out = mDict['out']
    optData = mDict['data']
    optModel = mDict['model']
    optLoss = mDict['loss']
    optTrain = mDict['train']

    # data
    df, x, y, c = loadData(optData)

    if type(x) is tuple:
        nx = x[0].shape[-1] + c.shape[-1]
    else:
        nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]

    # loss
    if eval(optLoss['name']) is hydroDL.model.crit.SigmaLoss:
        lossFun = hydroDL.model.crit.SigmaLoss(prior=optLoss['prior'])
        optModel['ny'] = ny * 2
    elif eval(optLoss['name']) is hydroDL.model.crit.RmseLoss:
        lossFun = hydroDL.model.crit.RmseLoss()
        optModel['ny'] = ny
    elif eval(optLoss['name']) is hydroDL.model.crit.NSELoss:
        lossFun = hydroDL.model.crit.NSELoss()
        optModel['ny'] = ny
    elif eval(optLoss['name']) is hydroDL.model.crit.NSELosstest:
        lossFun = hydroDL.model.crit.NSELosstest()
        optModel['ny'] = ny
    elif eval(optLoss['name']) is hydroDL.model.crit.MSELoss:
        lossFun = hydroDL.model.crit.MSELoss()
        optModel['ny'] = ny

    # model
    if optModel['nx'] != nx:
        print('updated nx by input data')
        optModel['nx'] = nx
    if eval(optModel['name']) is hydroDL.model.rnn.CudnnLstmModel:
        model = hydroDL.model.rnn.CudnnLstmModel(
            nx=optModel['nx'], ny=optModel['ny'], hiddenSize=optModel['hiddenSize']
        )
    elif eval(optModel['name']) is hydroDL.model.rnn.LstmCloseModel:
        model = hydroDL.model.rnn.LstmCloseModel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            hiddenSize=optModel['hiddenSize'],
            fillObs=True,
        )
    elif eval(optModel['name']) is hydroDL.model.rnn.AnnModel:
        model = hydroDL.model.rnn.AnnCloseModel(
            nx=optModel['nx'], ny=optModel['ny'], hiddenSize=optModel['hiddenSize']
        )
    elif eval(optModel['name']) is hydroDL.model.rnn.AnnCloseModel:
        model = hydroDL.model.rnn.AnnCloseModel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            hiddenSize=optModel['hiddenSize'],
            fillObs=True,
        )

    # train
    if optTrain['saveEpoch'] > optTrain['nEpoch']:
        optTrain['saveEpoch'] = optTrain['nEpoch']

    # train model
    writeMasterFile(mDict)
    model = hydroDL.model.train.trainModel(
        model,
        x,
        y,
        c,
        lossFun,
        nEpoch=optTrain['nEpoch'],
        miniBatch=optTrain['miniBatch'],
        saveEpoch=optTrain['saveEpoch'],
        saveFolder=out,
    )


def test(
    out,
    *,
    tRange,
    subset,
    doMC=False,
    suffix=None,
    batchSize=None,
    epoch=None,
    reTest=False,
    basinnorm=False
):
    mDict = readMasterFile(out)

    optData = mDict['data']
    if not os.path.isdir(optData['rootDB']):
        optData['rootDB'] = fixRootDB(optData['rootDB'])
    optData['subset'] = subset
    optData['tRange'] = tRange
    if 'damean' not in optData.keys():
        optData['damean'] = False
    if 'dameanopt' not in optData.keys():
        optData['dameanopt'] = 0
    if 'davar' not in optData.keys():
        optData['davar'] = 'streamflow'
    elif type(optData['davar']) is list:
        optData['davar'] = "".join(optData['davar'])

    # generate file names and run model
    filePathLst = namePred(out, tRange, subset, epoch=epoch, doMC=doMC, suffix=suffix)
    print('output files:', filePathLst)
    for filePath in filePathLst:
        if not os.path.isfile(filePath):
            reTest = True
    if reTest is True:
        print('Runing new results')
        df, x, obs, c = loadData(optData)
        model = loadModel(out, epoch=epoch)
        hydroDL.model.train.testModel(
            model, x, c, batchSize=batchSize, filePathLst=filePathLst, doMC=doMC
        )
    else:
        print('Loaded previous results')
        df, x, obs, c = loadData(optData, readX=False)

    # load previous result - readPred
    mDict = readMasterFile(out)
    dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
    for k in range(len(filePathLst)):
        filePath = filePathLst[k]
        dataPred[:, :, k] = pd.read_csv(filePath, dtype=np.float, header=None).values
    isSigmaX = False
    if mDict['loss']['name'] == 'hydroDL.model.crit.SigmaLoss' or doMC is not False:
        isSigmaX = True
        pred = dataPred[:, :, ::2]
        sigmaX = dataPred[:, :, 1::2]
    else:
        pred = dataPred

    if optData['doNorm'][1] is True:
        if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
            target = optData['target']
            if type(optData['target']) is not list:
                target = [target]
            pred = df.transform(pred, fieldLst=target, toNorm=False)
            obs = df.transform(obs, fieldLst=target, toNorm=False)
            if isSigmaX is True:
                sigmaX = df.transform(
                    sigmaX, fieldLst=target, toNorm=False, opt='sigma'
                )
        elif eval(optData['name']) is hydroDL.data.camels.DataframeCamels:
            pred = hydroDL.data.camels.transNorm(pred, 'usgsFlow', toNorm=False)
            obs = hydroDL.data.camels.transNorm(obs, 'usgsFlow', toNorm=False)
        if basinnorm is True:
            if type(subset) is list:
                gageid = np.array(subset)
            elif type(subset) is str:
                gageid = subset
            pred = hydroDL.data.camels.basinNorm(pred, gageid=gageid, toNorm=False)
            obs = hydroDL.data.camels.basinNorm(obs, gageid=gageid, toNorm=False)
    if isSigmaX is True:
        return df, pred, obs, sigmaX
    else:
        return df, pred, obs
