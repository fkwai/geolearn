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

# dataName = 'G200'

dataName='rmTK-B200'
DF = dbBasin.DataFrameBasin(dataName)
siteNoLst = DF.siteNoLst
codeLst = DF.varC
sn = 1e-5
predLst = [['logQ', 'sinT', 'cosT'], ['sinT', 'cosT'], ['logQ']]
labelLst = ['QS', 'S', 'Q']
dirLR = os.path.join(kPath.dirWQ, 'modelStat', 'LR-All')

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
dfT = pd.DataFrame({'date': DF.t}).set_index('date')
yr = dfT.index.year.values
t = yr+dfT.index.dayofyear.values/365
logQ=np.log(DF.q[:,:,1]+sn)
sinT = np.sin(2*np.pi*t)
cosT = np.cos(2*np.pi*t)

t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(kk, len(siteNoLst), time.time()-t0))
    # prep data
    varQ = 'runoff'
    varLst = DF.varC+[varQ]
    dfX = pd.DataFrame({'date': DF.t}).set_index('date')
    dfX['logQ']=logQ[:,kk]
    dfX['sinT'] = sinT
    dfX['cosT'] = cosT
    dfYP = pd.DataFrame(index=DF.t, columns=codeLst, dtype=float)
    dfYP.index.name = 'date'
    # for pred, label in zip(predLst, labelLst):
    pred=predLst[0]
    label=labelLst[0]
    saveName = os.path.join(dirLR, label, 'output', siteNo)
    for ic,code in enumerate(codeLst):
        x = dfX[pred].values            
        y=DF.c[:,kk,ic]
        [xx, yy], iv = utils.rmNan([x, y])
        if len(xx) > 10:
            xx = sm.add_constant(xx)
            model = sm.OLS(yy, xx).fit()
            yp = model.predict(sm.add_constant(x))
            dfYP[code] = yp
            dfpar = dictPar[label+'_'+code]
            dfpar.loc[siteNo][['b']+pred] = model.params
            dfpar.at[siteNo, 'count'] = len(xx)
            dfpar.at[siteNo, 'rsq'] = model.rsquared
        dfYP.to_csv(saveName)
