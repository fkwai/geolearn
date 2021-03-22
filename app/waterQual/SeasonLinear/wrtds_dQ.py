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
codeLst = usgs.newC

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
t0 = time.time()

dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-DQ', 'All')
dirOut = os.path.join(dirRoot, 'output')
dirPar = os.path.join(dirRoot, 'params')
for folder in [dirRoot, dirOut, dirPar]:
    if not os.path.exists(folder):
        os.mkdir(folder)

colLst = ['count', 'pSinT', 'pCosT', 'b']
dfPar = pd.DataFrame(index=siteNoLst, columns=colLst)

for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0))
    saveName = os.path.join(dirOut, siteNo)
    # if os.path.exists(saveName):
    #     continue
    df = waterQuality.readSiteTS(siteNo, varLst=['00060'], freq='D')
    dfX = pd.DataFrame({'date': df.index}).set_index('date')
    yr = dfX.index.year.values
    t = yr+dfX.index.dayofyear.values/365
    dfX['sinT'] = np.sin(2*np.pi*t)
    dfX['cosT'] = np.cos(2*np.pi*t)
    x = dfX.values
    y = np.log(df['00060'].values+sn)
    [xx, yy], iv = utils.rmNan([x, y])
    if len(xx) > 0:
        lrModel = LinearRegression()
        lrModel = lrModel.fit(xx, yy)
        yp = lrModel.predict(dfX.values)
        # yp = np.exp(yp)-sn
        dfYP = pd.DataFrame(index=df.index, columns=[
                            '00060'], data=np.exp(yp)-1)
        coef = lrModel.coef_
        inte = lrModel.intercept_
        parLst = [len(yy), coef[0], coef[1],  inte]
        dfPar.loc[siteNo] = parLst
    dfYP.to_csv(saveName)
filePar = os.path.join(dirPar, '00060')
dfPar.to_csv(filePar)
