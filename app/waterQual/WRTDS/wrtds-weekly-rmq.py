import importlib
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.post import axplot, figplot
from sklearn.linear_model import LinearRegression
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from scipy import stats
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# count sites
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2020, 1, 1)
sn = 1

# initial files
codeLst = sorted(usgs.codeLst)
siteNoLst = siteNoLstAll
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly_rmq')
dirOut = os.path.join(dirRoot, 'output')
dirRes = os.path.join(dirRoot, 'result')
if not os.path.exists(dirOut):
    os.mkdir(dirOut)
if not os.path.exists(dirRes):
    os.mkdir(dirRes)

# for testing
# codeLst = codeLst[:2]
# siteNoLst = siteNoLst[:10]

dictRes = dict()
colLst = ['count', 'pSinT', 'pCosT', 'b', 'corr', 'rmse']
for code in codeLst:
    dfRes = pd.DataFrame(index=siteNoLst, columns=colLst)
    dfRes.index.name = 'siteNo'
    dictRes[code] = dfRes

t0 = time.time()
for iS, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(iS, len(siteNoLst), time.time()-t0))
    varC = codeLst
    # varQ = ['00060']
    df = waterQuality.readSiteTS(siteNo, varLst=varC, freq='W')
    dfX = pd.DataFrame({'date': df.index}).set_index('date')
    yr = dfX.index.year.values
    t = dfX.index.dayofyear.values/365
    dfX['sinT'] = np.sin(2*np.pi*t)
    dfX['cosT'] = np.cos(2*np.pi*t)
    dfYP = pd.DataFrame(index=df.index, columns=varC)
    dfYP.index.name = 'date'
    for code in varC:
        # print(code)
        [xx, yy], _ = utils.rmNan([dfX.values, df[code].values])
        [xp], iv = utils.rmNan([dfX.values])
        if len(yy) <= 2:
            dictRes[code].loc[siteNo] = [len(yy)]+[np.nan for x in range(5)]
        else:
            lrModel = LinearRegression()
            lrModel = lrModel.fit(xx, yy)
            yp = lrModel.predict(xp)
            yt = lrModel.predict(xx)
            dfYP.at[dfX.index[iv], code] = yp
            coef = lrModel.coef_
            inte = lrModel.intercept_
            rmse = np.sqrt(np.nanmean((yt-yy)**2))
            if len(np.unique(yy)) == 1:
                corr = -9999
            else:
                corr = np.corrcoef(yt, yy)[0, 1]
            resLst = [len(yy), coef[0], coef[1], inte, corr, rmse]
            dictRes[code].loc[siteNo] = resLst
    dfYP.to_csv(os.path.join(dirOut, siteNo))

for code in codeLst:
    fileRes = os.path.join(dirRes, code)
    dictRes[code].to_csv(fileRes)
