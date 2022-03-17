import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import statsmodels.api as sm
import time

dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)
siteNoLst = DF.siteNoLst
codeLst = DF.varC
sn = 1e-5
predLst = [['logQ', 'sinT', 'cosT'], ['sinT', 'cosT'], ['logQ']]
labelLst = ['QS', 'S', 'Q']
dirLR = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All'


# dictPar contains all saved par and rsq
colLst = ['count', 'rsq', 'b']
dictPar = dict()
for pred, label in zip(predLst, labelLst):
    dirPar = os.path.join(dirLR, label, 'param')
    dirOut = os.path.join(dirLR, label, 'output')
    if not os.path.isdir(dirPar):
        os.makedirs(dirPar)
    if not os.path.isdir(dirOut):
        os.makedirs(dirOut)
    for code in codeLst:
        dfpar = pd.DataFrame(index=siteNoLst, columns=colLst+pred)
        dfpar.index.name = 'siteNo'
        dictPar[label+'_'+code] = dfpar

# start regression
t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(kk, len(siteNoLst), time.time()-t0))
    # prep data
    varQ = '00060'
    varLst = DF.varC+[varQ]
    df = dbBasin.io.readSiteTS(siteNo, varLst=varLst, freq='D')
    dfX = pd.DataFrame({'date': df.index}).set_index('date')
    yr = dfX.index.year.values
    t = yr+dfX.index.dayofyear.values/365
    dfX = dfX.join(np.log(df[varQ]+sn)).rename(
        columns={varQ: 'logQ'})
    dfX['sinT'] = np.sin(2*np.pi*t)
    dfX['cosT'] = np.cos(2*np.pi*t)
    dfYP = pd.DataFrame(index=df.index, columns=codeLst, dtype=np.float)
    dfYP.index.name = 'date'
    for pred, label in zip(predLst, labelLst):
        saveName = os.path.join(dirLR, 'label', 'output', siteNo)
        for code in codeLst:
            x = dfX[pred].values
            y = df[code].values
            [xx, yy], iv = utils.rmNan([x, y])
            if len(xx) > 10:
                xx = sm.add_constant(xx)
                model = sm.OLS(yy, xx).fit()
                yp = model.predict(sm.add_constant(x))
                dfYP[code] = yp
                dfpar = dictPar[label+'_'+code]
                dfpar.at[siteNo, ['b']+pred] = model.params
                dfpar.at[siteNo, 'count'] = len(xx)
                dfpar.at[siteNo, 'rsq'] = model.rsquared
            dfYP.to_csv(saveName)
