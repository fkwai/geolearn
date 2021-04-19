from joblib import Parallel, delayed
import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm

startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2020, 1, 1)
sn = 1
codeLst = usgs.newC

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
t0 = time.time()

dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W',)
dirOut = os.path.join(dirRoot, 'B10')
for folder in [dirRoot, dirOut]:
    if not os.path.exists(folder):
        os.mkdir(folder)


def func(siteNo, fitAll=True):
    # prep data
    print(siteNo)
    saveName = os.path.join(dirOut, siteNo)
    if os.path.exists(saveName):
        return()
    t0 = time.time()
    varQ = '00060'
    varLst = codeLst+[varQ]
    df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq='W')
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
    # train / test
    fitCodeLst = list()
    for code in codeLst:
        if siteNo in dictSite[code]:
            fitCodeLst.append(code)
    for code in fitCodeLst:
        ind1 = np.where(yr < 2010)[0]
        ind2 = np.where(yr >= 2010)[0]
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
        print(siteNoLst.index(siteNo), siteNo, code, t1-t0)
    saveName = os.path.join(dirOut, siteNo)
    dfYP.to_csv(saveName)
    return


results = Parallel(n_jobs=-1)(delayed(func)(siteNo) for siteNo in siteNoLst)
