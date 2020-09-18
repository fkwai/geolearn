import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
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

wqData = waterQuality.DataModelWQ('sbWT')

siteNoLst = wqData.info['siteNo'].unique().tolist()
# trainSetLst = ['Y1', 'Y2']

trainSet = 'Y1'
dfCorrLst = [pd.DataFrame(index=siteNoLst, columns=usgs.varC)
             for x in range(2)]
dfRmseLst = [pd.DataFrame(index=siteNoLst, columns=usgs.varC)
             for x in range(2)]
t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0))
    outFolder = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-F')
    saveFile = os.path.join(outFolder, trainSet, siteNo)
    dfP = pd.read_csv(saveFile, index_col=None).set_index('date')
    dfP.index = pd.to_datetime(dfP.index)
    dfC = waterQuality.readSiteTS(siteNo, varLst=usgs.varC, freq='W')
    yr = dfC.index.year.values
    indLst = [np.where(yr % 2 == x)[0] for x in [1, 0]]
    for code in usgs.varC:
        for k in range(2):
            ind = indLst[k]
            corr = dfC.iloc[ind][code].corr(dfP.iloc[ind][code])
            rmse = np.sqrt(
                np.sum((dfC.iloc[ind][code]-dfP.iloc[ind][code])**2))
            dfCorrLst[k].loc[siteNo][code] = corr
            dfRmseLst[k].loc[siteNo][code] = corr
for k in range(2):
    if k == 0:
        testSet = 'Y2'
    else:
        testSet = 'Y1'
    dfCorrLst[k].to_csv(os.path.join(
        outFolder, '{}-{}-corr'.format(trainSet, testSet)))
    dfRmseLst[k].to_csv(os.path.join(
        outFolder, '{}-{}-rmse'.format(trainSet, testSet)))

siteNo = '01013500'
code = '00010'
dfP = pd.read_csv(saveFile, index_col=None).set_index('date')
dfP.index = pd.to_datetime(dfP.index)

dfC = waterQuality.readSiteTS(siteNo, varLst=usgs.varC, freq='W')
fig, ax = plt.subplots(1, 1)
ax.plot(dfC[code], '*r')
ax.plot(dfP[code], '-b')
fig.show()
