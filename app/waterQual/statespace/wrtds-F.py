import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.post import axplot, figplot
from sklearn.linear_model import LinearRegression


import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2020, 1, 1)
sn = 0.001

wqData = waterQuality.DataModelWQ('basinRef', rmFlag=True)
siteNoLst = wqData.siteNoLst
t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0), end='\r')
    saveLst = [os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-F', x, siteNo)
               for x in ['Yeven', 'Yodd']]
    if os.path.exists(saveLst[0]) and os.path.exists(saveLst[1]):
        continue
    dfC, dfCF = usgs.readSample(
        siteNo, codeLst=usgs.varC, startDate=startDate, flag=2)
    dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
    dfF = gridMET.readBasin(siteNo)
    yr = dfC.index.year.values
    indLst = [np.where(yr % 2 == x)[0] for x in [0, 1]]
    dfX = pd.DataFrame({'date': dfC.index}).set_index('date')
    dfY = np.log(dfC+sn)
    dfX = dfX.join(np.log(dfQ['00060_00003']+sn)).rename(
        columns={'00060_00003': 'logQ'})
    dfX=dfX.join(dfF)
    yr = dfX.index.year.values
    t = yr+dfX.index.dayofyear.values/365
    # dfX['t'] = t-1979
    dfX['sinT'] = np.sin(2*np.pi*t)
    dfX['cosT'] = np.cos(2*np.pi*t)

    ctR = pd.date_range(startDate, endDate)
    dfXP = pd.DataFrame({'date': ctR}).set_index('date')
    dfXP = dfXP.join(np.log(dfQ['00060_00003']+0.01)).rename(
        columns={'00060_00003': 'logQ'})
    dfXP=dfXP.join(dfF)
    yr = dfXP.index.year.values
    t = yr+dfXP.index.dayofyear.values/365
    # dfXP['t'] = t-1979
    dfXP['sinT'] = np.sin(2*np.pi*t)
    dfXP['cosT'] = np.cos(2*np.pi*t)

    for k in range(2):
        ind = indLst[k]
        saveFile = saveLst[k]
        dfYP = pd.DataFrame(index=ctR, columns=usgs.varC)
        dfYP.index.name = 'date'
        if len(ind) > 0:
            for code in usgs.varC:
                [xx, yy], iv = utils.rmNan(
                    [dfX.iloc[ind].values, dfY.iloc[ind][code].values])
                if len(xx) > 0:
                    lrModel = LinearRegression()
                    lrModel = lrModel.fit(xx, yy)
                    b = dfXP.isna().any(axis=1)
                    yp = lrModel.predict(dfXP[~b].values)
                    yp = np.exp(yp)-sn
                    dfYP.at[dfYP[~b].index, code] = yp
        dfYP.to_csv(saveFile)
