import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
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



dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-all')
dirOut = os.path.join(dirRoot, 'output')
dirPar = os.path.join(dirRoot, 'parameter')
if not os.path.exists(dirOut):
    os.mkdir(dirOut)
if not os.path.exists(dirPar):
    os.mkdir(dirPar)

codeLst=sorted(usgs.codeLst)

startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2020, 1, 1)
sn = 1

t0 = time.time()
saveDir = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-all')
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
# for kk, siteNo in enumerate(siteNoLst):

siteNo = siteNoLst[0]

print('{}/{} {:.2f}'.format(
    kk, len(siteNoLst), time.time()-t0))
saveName = os.path.join(saveFolder, siteNo)
# if os.path.exists(saveName):
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
dfYP = pd.DataFrame(index=df.index, columns=varC)
dfYP.index.name = 'date'
# for code in varC:

code = '00955'
x = dfX.values
y = df[code].values
[xx, yy], iv = utils.rmNan([x, y])
if len(iv) > 0:
    lrModel = LinearRegression()
    lrModel = lrModel.fit(xx, yy)
    b = dfX.isna().any(axis=1)
    yp = lrModel.predict(dfX[~b].values)
    yp = lrModel.predict(dfXN[~b].values)
    # yp = np.exp(yp)-sn
    dfYP.at[dfYP[~b].index, code] = yp
dfYP.to_csv(saveName)


y1 = lrModel.predict(dfX[~b].values)
y2 = lrModel.predict(xx)


coef = lrModel.coef_
inte = lrModel.intercept_
xx.dot(coef)+inte
yp
