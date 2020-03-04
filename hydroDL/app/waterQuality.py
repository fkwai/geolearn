import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET

fileCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'codeWQ.csv')
codePdf = pd.read_csv(fileCode, dtype=str).set_index('code')
codeLst = list(codePdf.index)


def wrapData(caseName, siteNoLst, *, rho=365, nFill=5, varC=codeLst, varG=gageII.lstWaterQuality):
    """ wrap up input and target data for the model,as:
    x=[nT,nP,nX]
    y=[nP,nY]
    c=[nP,nC]
    where nP is number of time series
    Arguments:
        caseName {str} -- name of current data case
        siteNoLst {list} -- list of USGS site
    Keyword Arguments:
        rho {int} -- [description] (default: {365})
        nFill {int} -- max number of continous nan to interpolate in input data (default: {5})
        varC {list} -- list of water quality code to learn (default: {usgs.lstCodeSample})
        varG {list} -- list of constant variables in gageII (default: {gageII.lstWaterQuality})
        varQ and varF are fixed so far
    """
    # add a start date to improve efficiency.
    startDate = pd.datetime(1979, 1, 1)

    # gageII
    tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)
    tabG = gageII.updateCode(tabG)

    # read data and merge to: f/q=[nT,nP,nX], g/c=[nP,nY]
    fLst = list()  # forcing ts
    gLst = list()  # geo-const
    qLst = list()  # streamflow
    cLst = list()  # water quality
    infoLst = list()
    t0 = time.time()
    for i, siteNo in enumerate(siteNoLst):
        t1 = time.time()
        dfC = usgs.readSample(siteNo, codeLst=varC, startDate=startDate)
        dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
        dfF = gridMET.readBasin(siteNo)        
        for k in range(len(dfC)):
            ct = dfC.index[k]
            ctR = pd.date_range(ct-pd.Timedelta(days=rho-1), ct)
            if ctR[0] < startDate:
                continue
            tempQ = pd.DataFrame({'date': ctR}).set_index('date').join(
                dfQ).interpolate(limit=nFill, limit_direction='both')
            tempF = pd.DataFrame({'date': ctR}).set_index('date').join(
                dfF).interpolate(limit=nFill, limit_direction='both')
            qLst.append(tempQ.values)
            fLst.append(tempF.values)
            cLst.append(dfC.iloc[k].values)
            gLst.append(tabG.loc[siteNo].values)
            infoLst.append(dict(siteNo=siteNo, date=ct))
        t2 = time.time()
        print('{} on site {} reading {:.3} total {}'.format(
            i, siteNo, t2-t1, t2-t0))
    q = np.stack(qLst, axis=-1).swapaxes(1, 2).astype(np.float64)
    f = np.stack(fLst, axis=-1).swapaxes(1, 2).astype(np.float64)
    g = np.stack(gLst, axis=-1).swapaxes(0, 1).astype(np.float64)
    c = np.stack(cLst, axis=-1).swapaxes(0, 1).astype(np.float64)
    infoDf = pd.DataFrame(infoLst)

    saveFolder = os.path.join(kPath.dirWQ, 'trainData')
    saveName = os.path.join(saveFolder, caseName)
    np.savez(saveName, q=q, f=f, c=c, g=g)
    infoDf.to_csv(saveName+'.csv')
    dictData = dict(name=caseName, rho=rho, nFill=nFill,
                    varG=varG, varC=varC, siteNoLst=siteNoLst)
    with open(saveName+'.json', 'w') as fp:
        json.dump(dictData, fp, indent=4)


def loadData(caseName):
    saveName = os.path.join(kPath.dirWQ, 'trainData', caseName)
    npzFile = np.load(saveName+'.npz')
    q = npzFile['q']
    f = npzFile['f']
    c = npzFile['c']
    g = npzFile['g']
    info = pd.read_csv(saveName+'.csv', index_col=0, dtype={'siteNo': str})
    with open(saveName+'.json', 'r') as fp:
        dictData = json.load(fp)
    return dictData, info, q, c, f, g


def loadInfo(caseName):
    saveName = os.path.join(kPath.dirWQ, 'trainData', caseName)
    info = pd.read_csv(saveName+'.csv', index_col=0, dtype={'siteNo': str})
    with open(saveName+'.json', 'r') as fp:
        dictData = json.load(fp)
    return dictData, info


def divideTrain(info, ratio=0.8):
    # devide training and testing - last 20% as testing
    siteNoLst = info['siteNo'].unique().tolist()
    trainIndLst = list()
    testIndLst = list()
    for siteNo in siteNoLst:
        indLst = info[info['siteNo'] == siteNo].index.tolist()
        indSep = round(len(indLst)*0.8)
        trainIndLst = trainIndLst+indLst[:indSep]
        testIndLst = testIndLst+indLst[indSep:]
    return trainIndLst, testIndLst
