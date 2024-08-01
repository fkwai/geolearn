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
predLst = [['logQ', 'sinT', 'cosT'], ['sinT', 'cosT'], ['logQ'],['logQ', 'sinT', 'cosT','t']]
labelLst = ['QS', 'S', 'Q','QST']
# dirLR = r'C:\Users\geofk\work\waterQuality\modelStat\LR-log'
dirLR = r'C:\Users\geofk\work\waterQuality\modelStat\LR'

pred = predLst[0]
label = labelLst[0]

# dictPar contains all saved par and rsq
colLst = ['count', 'rsq', 'b']
dictPar = dict()
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
yr=pd.to_datetime(DF.t).year
yrD=pd.to_datetime(DF.t).dayofyear
t = yr+yrD/365
sinT=np.sin(2*np.pi*t)
cosT=np.cos(2*np.pi*t)

siteNo=siteNoLst[0]


t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(kk, len(siteNoLst), time.time()-t0))
    # prep data
    varQ = '00060'
    indS=siteNoLst.index(siteNo)
    Q=DF.q[:,indS,0]
    logQ=np.log(Q+sn)
    dfX = pd.DataFrame({'logQ': logQ,'cosT':cosT,'sinT':sinT,'t':t})    
    # dfYP = pd.DataFrame(index=DF.t, columns=codeLst, dtype=np.float)
    # dfYP.index.name = 'date'
    saveName = os.path.join(dirLR, label, 'output', siteNo)
    for code in codeLst:
        x = dfX[pred].values
        indC=DF.varC.index(code)
        y = DF.c[:, indS, indC]
        [xx, yy], iv = utils.rmNan([x, y])
        if len(xx) > 10:
            xx = sm.add_constant(xx)
            model = sm.OLS(yy, xx).fit()
            yp = model.predict(sm.add_constant(x))
            # dfYP[code] = yp
            dfpar = dictPar[label+'_'+code]
            dfpar.at[siteNo, ['b']+pred] = model.params
            dfpar.at[siteNo, 'count'] = len(xx)
            dfpar.at[siteNo, 'rsq'] = model.rsquared
        # dfYP.to_csv(saveName)

dirPar = os.path.join(dirLR, label, 'param')
for code in codeLst:
    filePar = os.path.join(dirPar, code)
    dictPar[label+'_'+code].to_csv(filePar)