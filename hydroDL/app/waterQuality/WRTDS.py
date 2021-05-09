import os
import json
import numpy as np
import pandas as pd
import time
from hydroDL import kPath, utils
from hydroDL.data import usgs, transform, dbBasin
import statsmodels.api as sm

sn = 1


def loadWRTDS(siteNo, freq, trainSet='B10', the=[150, 50], codeLst=usgs.varC):
    dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-{}'.format(freq))
    dirOut = os.path.join(dirRoot, trainSet)
    saveName = os.path.join(dirOut, siteNo+'.csv')
    if os.path.exists(saveName):
        dfW = pd.read_csv(saveName, index_col=None).set_index('date')
    else:
        print('do calWRTDS before')
        dfW = calWRTDS(siteNo, freq, trainSet=trainSet,
                       the=the, codeLst=usgs.varC)
    return dfW


def calWRTDS(siteNo, freq, trainSet='B10', the=[150, 50], fitAll=True, codeLst=usgs.varC, reCal=False):
    dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-{}'.format(freq))
    dirOut = os.path.join(dirRoot, trainSet)
    saveName = os.path.join(dirOut, siteNo)
    if os.path.exists(saveName):
        print('calculated {}'.format(siteNo))
        if reCal is False:
            return
    t0 = time.time()
    varQ = '00060'
    varLst = codeLst+[varQ]
    df = dbBasin.readSiteTS(siteNo, varLst=varLst, freq=freq)
    dfYP = pd.DataFrame(index=df.index, columns=codeLst)
    dfX = pd.DataFrame({'date': df.index}).set_index('date')
    dfX = dfX.join(np.log(df[varQ]+sn)).rename(
        columns={varQ: 'logQ'})
    yr = dfX.index.year.values
    t = yr+dfX.index.dayofyear.values/365
    dfX['sinT'] = np.sin(2*np.pi*t)
    dfX['cosT'] = np.cos(2*np.pi*t)
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
        dfXY = dfX.join(np.log(df[code]+sn))
        df1 = dfXY.iloc[ind1].dropna()
        if fitAll:
            df2 = dfXY[xVarLst+['t']].dropna()
        else:
            df2 = dfXY.iloc[ind2].dropna()  # only fit for observations now
        n = len(df1)
        if n == 0:
            break
        # calculate weight
        h = np.array([7, 2, 0.5])  # window [Y Q S] from EGRET
        tLst = df2.index.tolist()
        for t in tLst:
            dY = np.abs((df2.loc[t]['t']-df1['t']).values)
            dQ = np.abs((df2.loc[t]['logQ']-df1['logQ']).values)
            dS = np.min(
                np.stack([abs(np.ceil(dY)-dY), abs(dY-np.floor(dY))]), axis=0)
            d = np.stack([dY, dQ, dS])
            if n > 100:
                hh = np.repeat(h[:, None], n, axis=1)
                bW = False
                while ~bW:
                    bW = np.min(np.sum((hh-d) > 0, axis=1)) > 100
                    hh = hh*1.1 if not bW else hh
            else:
                htemp = np.max(d, axis=1)*1.1
                hh = np.repeat(htemp[:, None], n, axis=1)
            w = (1-(d/hh)**3)**3
            w[w < 0] = 0
            wAll = w[0]*w[1]*w[2]
            ind = np.where(wAll > 0)[0]
            ww = wAll[ind]
            # fit WLS
            Y = df1.iloc[ind][code].values
            X = df1.iloc[ind][xVarLst].values
            model = sm.WLS(Y, X, weights=ww).fit()
            xp = df2.loc[t][xVarLst].values
            yp = model.predict(xp)[0]
            dfYP.loc[t][code] = np.exp(yp)-sn
        t1 = time.time()
        print(k, siteNo, code, t1-t0)
    saveName = os.path.join(dirOut, siteNo)
    dfYP.to_csv(saveName)
    return dfYP


def defineTrainSet(t, trainSet):
    if trainSet == 'B10':
        yr = t.year.values
        ind1 = np.where(yr < 2010)[0]
        ind2 = np.where(yr >= 2010)[0]
    return ind1, ind2
