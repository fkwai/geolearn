import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from sklearn.linear_model import LinearRegression
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2020, 1, 1)
sn = 1

# rmCode = ['00010', '00095', '00400']
# rmName = 'rmTKH'
rmCode = ['00010', '00095']
rmName = 'rmTK'
# rmCode = ['00010']
# rmName = 'rmT'

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N2.json')) as f:
    dictSite = json.load(f)
codeLst = sorted(list(set(usgs.newC)-set(rmCode)))
siteNoLst = dictSite[rmName]

dirRoot = os.path.join(kPath.dirWQ, 'modelStat',
                       'WRTDS-W', 'B10-{}'.format(rmName))
dirOut = os.path.join(dirRoot, 'output')
dirPar = os.path.join(dirRoot, 'params')
for folder in [dirRoot, dirOut, dirPar]:
    if not os.path.exists(folder):
        os.mkdir(folder)

t0 = time.time()
for kk, siteNo in enumerate(siteNoLst[:1]):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0))
    saveName = os.path.join(dirOut, siteNo)
    # if os.path.exists(saveName):
    #     continue
    varQ = '00060'
    varLst = usgs.newC+[varQ]
    df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq='W')

    dfX = pd.DataFrame({'date': df.index}).set_index('date')
    dfX = dfX.join(np.log(df[varQ]+sn)).rename(
        columns={varQ: 'logQ'})
    for code in rmCode:
        dfX = dfX.join(df[code])
    yr = dfX.index.year.values
    t = yr+dfX.index.dayofyear.values/365
    dfX['sinT'] = np.sin(2*np.pi*t)
    dfX['cosT'] = np.cos(2*np.pi*t)
    ind = np.where(yr < 2010)[0]
    dfYP = pd.DataFrame(index=df.index, columns=codeLst)
    dfYP.index.name = 'date'
    # dfXN = (dfX-dfX.min())/(dfX.max()-dfX.min())
    dfXN = dfX
    for code in codeLst:
        x = dfXN.iloc[ind].values
        # y = np.log(df.iloc[ind][code].values+sn)
        y = df.iloc[ind][code].values
        [xx, yy], iv = utils.rmNan([x, y])
        if len(yy) > 0:
            lrModel = LinearRegression()
            lrModel = lrModel.fit(xx, yy)
            b = dfXN.isna().any(axis=1)
            yp = lrModel.predict(dfXN[~b].values)
            # yp = np.exp(yp)-sn
            dfYP.at[dfYP[~b].index, code] = yp
    dfYP.to_csv(saveName)
