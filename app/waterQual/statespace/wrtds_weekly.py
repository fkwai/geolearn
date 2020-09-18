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

wqData = waterQuality.DataModelWQ('sbWT')
siteNoLst = wqData.siteNoLst
t0 = time.time()
addF = True
for addF in [True, False]:
    for kk, siteNo in enumerate(siteNoLst):
        print('{}/{} {:.2f}'.format(
            kk, len(siteNoLst), time.time()-t0))
        if addF is True:
            saveLst = [os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-F', x, siteNo)
                       for x in ['Y1', 'Y2']]
        else:
            saveLst = [os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS', x, siteNo)
                       for x in ['Y1', 'Y2']]
        # if os.path.exists(saveLst[0]) and os.path.exists(saveLst[1]):
        #     continue
        varF = gridMET.varLst+ntn.varLst
        varC = usgs.varC
        varQ = usgs.varQ
        varLst = varF+varC+varQ
        df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq='W')

        dfX = pd.DataFrame({'date': df.index}).set_index('date')
        dfX = dfX.join(np.log(df['00060']+sn)).rename(
            columns={'00060': 'logQ'})
        if addF is True:
            dfX = dfX.join(df[varF])
        yr = dfX.index.year.values
        t = yr+dfX.index.dayofyear.values/365
        dfX['sinT'] = np.sin(2*np.pi*t)
        dfX['cosT'] = np.cos(2*np.pi*t)
        indLst = [np.where(yr % 2 == x)[0] for x in [1, 0]]
        # for k in range(2):
        k = 0
        ind = indLst[k]
        saveFile = saveLst[k]
        dfYP = pd.DataFrame(index=df.index, columns=varC)
        dfYP.index.name = 'date'
        dfXN = (dfX-dfX.min())/(dfX.max()-dfX.min())
        for code in varC:
            x = dfXN.iloc[ind].values
            # y = np.log(df.iloc[ind][code].values+sn)
            y = df.iloc[ind][code].values
            [xx, yy], iv = utils.rmNan([x, y])
            if len(xx) > 0:
                lrModel = LinearRegression()
                lrModel = lrModel.fit(xx, yy)
                b = dfX.isna().any(axis=1)
                yp = lrModel.predict(dfX[~b].values)
                # yp = np.exp(yp)-sn
                dfYP.at[dfYP[~b].index, code] = yp
        dfYP.to_csv(saveFile)
