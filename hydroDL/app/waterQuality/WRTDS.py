import os
import json
import numpy as np
import pandas as pd
import time
from hydroDL import kPath, utils
from hydroDL.data import usgs, transform, dbBasin
import statsmodels.api as sm

sn = 1e-5


def loadSite(siteNo, freq='D', trainSet='B10', the=[150, 50], codeLst=usgs.varC):
    dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-{}'.format(freq))
    dirOut = os.path.join(dirRoot, trainSet)
    saveName = os.path.join(dirOut, siteNo + '.csv')
    if os.path.exists(saveName):
        dfW = pd.read_csv(saveName, index_col=None).set_index('date')
    else:
        print('do calWRTDS before')
        dfW = calWRTDS(siteNo, freq, trainSet=trainSet, the=the, codeLst=usgs.varC)
    return dfW[codeLst]


def loadMat(siteNoLst, codeLst, freq='D', trainSet='B10'):
    dfW = loadSite(siteNoLst[0])
    nt = len(dfW)
    out = np.ndarray([nt, len(siteNoLst), len(siteNoLst)])
    for indS, siteNo in enumerate(siteNoLst):
        for indC, code in enumerate(codeLst):
            dfW = loadSite(siteNo, freq=freq, trainSet=trainSet, codeLst=codeLst)
            out[:, indS, indC] = dfW[codeLst].values


def calWRTDS(
    siteNo,
    freq='D',
    trainSet='B10',
    the=[150, 50],
    fitAll=True,
    codeLst=usgs.varC,
    reCal=False,
):
    # legacy code
    dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-{}'.format(freq))
    dirOut = os.path.join(dirRoot, trainSet)
    saveName = os.path.join(dirOut, siteNo)
    if os.path.exists(saveName):
        print('calculated {}'.format(siteNo))
        if reCal is False:
            return
    t0 = time.time()
    varQ = '00060'
    varLst = codeLst + [varQ]
    df = dbBasin.readSiteTS(siteNo, varLst=varLst, freq=freq)
    dfYP = pd.DataFrame(index=df.index, columns=codeLst)
    dfX = pd.DataFrame({'date': df.index}).set_index('date')
    dfX = dfX.join(np.log(df[varQ] + sn)).rename(columns={varQ: 'logQ'})
    yr = dfX.index.year.values
    t = yr + dfX.index.dayofyear.values / 365
    dfX['sinT'] = np.sin(2 * np.pi * t)
    dfX['cosT'] = np.cos(2 * np.pi * t)
    dfX['yr'] = yr
    dfX['t'] = t
    xVarLst = ['yr', 'logQ', 'sinT', 'cosT']
    ind1, ind2 = defineTrainSet(df.index, trainSet)
    # train / test
    fitCodeLst = list()
    for code in codeLst:
        b1 = df.iloc[ind1][code].dropna().shape[0] > the[0]
        b2 = df.iloc[ind2][code].dropna().shape[0] > the[1]
        if b1 and b2:
            fitCodeLst.append(code)
    for code in fitCodeLst:
        dfXY = dfX.join(np.log(df[code] + sn))
        df1 = dfXY.iloc[ind1].dropna()
        if fitAll:
            df2 = dfXY[xVarLst + ['t']].dropna()
        else:
            df2 = dfXY.iloc[ind2].dropna()  # only fit for observations now
        n = len(df1)
        if n == 0:
            break
        # calculate weight
        h = np.array([7, 2, 0.5])  # window [Y Q S] from EGRET
        tLst = df2.index.tolist()
        for t in tLst:
            dY = np.abs((df2.loc[t]['t'] - df1['t']).values)
            dQ = np.abs((df2.loc[t]['logQ'] - df1['logQ']).values)
            dS = np.min(
                np.stack([abs(np.ceil(dY) - dY), abs(dY - np.floor(dY))]), axis=0
            )
            d = np.stack([dY, dQ, dS])
            ww, ind = calWeight(d)
            # fit WLS
            Y = df1.iloc[ind][code].values
            X = df1.iloc[ind][xVarLst].values
            model = sm.WLS(Y, X, weights=ww).fit()
            xp = df2.loc[t][xVarLst].values
            yp = model.predict(xp)[0]
            dfYP.loc[t][code] = np.exp(yp) - sn
        t1 = time.time()
        print(siteNo, code, t1 - t0)
    saveName = os.path.join(dirOut, siteNo)
    dfYP.to_csv(saveName)
    return dfYP


def defineTrainSet(t, trainSet):
    if trainSet == 'B10':
        yr = t.year.values
        ind1 = np.where(yr < 2010)[0]
        ind2 = np.where(yr >= 2010)[0]
    return ind1, ind2


def testWRTDS(dataName, trainSet, testSet, codeLst):
    DF = dbBasin.DataFrameBasin(dataName)
    dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
    dirName = '{}-{}-{}'.format(dataName, trainSet, testSet)
    outFolder = os.path.join(dirRoot, dirName)
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    # Calculate WRTDS from train and test set
    varX = ['streamflow']
    varY = codeLst
    d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varX=varX, varY=varY)
    d2 = dbBasin.DataModelBasin(DF, subset=testSet, varX=varX, varY=varY)
    tt1 = pd.to_datetime(d1.t)
    yr1 = tt1.year.values
    t1 = yr1 + tt1.dayofyear.values / 365
    sinT1 = np.sin(2 * np.pi * t1)
    cosT1 = np.cos(2 * np.pi * t1)
    tt2 = pd.to_datetime(d2.t)
    yr2 = tt2.year.values
    t2 = yr2 + tt2.dayofyear.values / 365
    sinT2 = np.sin(2 * np.pi * t2)
    cosT2 = np.cos(2 * np.pi * t2)
    ###
    yOut = np.full([len(d2.t), len(d2.siteNoLst), len(varY)], np.nan)
    t0 = time.time()
    for indS, siteNo in enumerate(d2.siteNoLst):
        siteFile = os.path.join(outFolder, siteNo)
        if os.path.exists(siteFile):
            yp = pd.read_csv(siteFile, index_col=0).values
            yOut[:, indS, :] = yp
            continue
        for indC, code in enumerate(varY):
            print('{} {} {} {}'.format(indS, siteNo, code, time.time() - t0))
            y1 = d1.Y[:, indS, indC].copy()
            q1 = d1.X[:, indS, 0].copy()
            q1[q1 < 0] = 0
            logq1 = np.log(q1 + sn)
            x1 = np.stack([logq1, yr1, sinT1, cosT1]).T
            y2 = d2.Y[:, indS, indC].copy()
            q2 = d2.X[:, indS, 0].copy()
            q2[q2 < 0] = 0
            logq2 = np.log(q2 + sn)
            x2 = np.stack([logq2, yr2, sinT2, cosT2]).T
            [xx1, yy1], ind1 = utils.rmNan([x1, y1])
            if testSet == 'all':
                [xx2], ind2 = utils.rmNan([x2])
            else:
                [xx2, yy2], ind2 = utils.rmNan([x2, y2])
            if len(ind1) < 40:
                continue
            for k in ind2:
                dY = np.abs(t2[k] - t1[ind1])
                dQ = np.abs(logq2[k] - logq1[ind1])
                dS = np.min(
                    np.stack([abs(np.ceil(dY) - dY), abs(dY - np.floor(dY))]), axis=0
                )
                d = np.stack([dY, dQ, dS])
                ww, ind = calWeight(d)
                model = sm.WLS(yy1[ind], xx1[ind], weights=ww).fit()
                yp = model.predict(x2[k, :])[0]
                yOut[k, indS, indC] = yp
        # save a siteFile
        dfOut = pd.DataFrame(index=d2.t, columns=codeLst, data=yOut[:, indS, :])
        dfOut.to_csv(siteFile)
    return yOut


def calWeight(d, h=[7, 2, 0.5], the=100):
    # window [Y Q S] from EGRET
    n = d.shape[1]
    if n > the:
        hh = np.tile(h, [n, 1]).T
        bW = False
        while ~bW:
            bW = np.sum(np.all(hh > d, axis=0)) > the
            hh = hh * 1.1 if not bW else hh
    else:
        htemp = np.max(d, axis=1) * 1.1
        hh = np.repeat(htemp[:, None], n, axis=1)
    w = (1 - (d / hh) ** 3) ** 3
    w[w < 0] = 0
    wAll = w[0] * w[1] * w[2]
    ind = np.where(wAll > 0)[0]
    ww = wAll[ind]
    return ww, ind
