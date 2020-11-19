import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time



# rmCode = ['00010', '00095', '00400']
# rmName = 'rmTKH'
# rmCode = ['00010', '00095']
# rmName = 'rmTK'
rmCode = ['00010']
rmName = 'rmT'

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
codeLst = sorted(list(set(usgs.newC)-set(rmCode)))
siteNoLst = dictSite[rmName]

trainSet = 'B10N5'
testSet = 'A10N5'
df = pd.DataFrame(index=siteNoLst, columns=usgs.newC)
df.index.name = 'siteNo'

dirRoot = os.path.join(kPath.dirWQ, 'modelStat',
                       'WRTDS-W', 'B10-{}'.format(rmName))
dirOut = os.path.join(dirRoot, 'output')
dirPar = os.path.join(dirRoot, 'params')

dfCorr1 = df.copy()
dfCorr2 = df.copy()
dfRmse1 = df.copy()
dfRmse2 = df.copy()
t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0))
    saveFile = os.path.join(dirOut, siteNo)
    dfP = pd.read_csv(saveFile, index_col=None).set_index('date')
    dfP.index = pd.to_datetime(dfP.index)
    dfC = waterQuality.readSiteTS(siteNo, varLst=usgs.newC, freq='W')
    yr = dfC.index.year.values
    for code in codeLst:
        ind1 = np.where(yr < 2010)[0]
        ind2 = np.where(yr >= 2010)[0]
        rmse1, corr1 = utils.stat.calErr(
            dfP.iloc[ind1][code].values,  dfC.iloc[ind1][code].values)
        rmse2, corr2 = utils.stat.calErr(
            dfP.iloc[ind2][code].values,  dfC.iloc[ind2][code].values)
        dfCorr1.loc[siteNo][code] = corr1
        dfRmse1.loc[siteNo][code] = rmse1
        dfCorr2.loc[siteNo][code] = corr2
        dfRmse2.loc[siteNo][code] = rmse2

dfCorr1.to_csv(os.path.join(
    dirRoot, '{}-{}-corr'.format(trainSet, trainSet)))
dfRmse1.to_csv(os.path.join(
    dirRoot, '{}-{}-rmse'.format(trainSet, trainSet)))
dfCorr2.to_csv(os.path.join(
    dirRoot, '{}-{}-corr'.format(trainSet, testSet)))
dfRmse2.to_csv(os.path.join(
    dirRoot, '{}-{}-rmse'.format(trainSet, testSet)))
