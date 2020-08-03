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

wqData = waterQuality.DataModelWQ('basinRef', rmFlag=True)

siteNoLst = wqData.info['siteNo'].unique().tolist()
trainSetLst = ['Yodd', 'Yeven']

for trainSet in trainSetLst:
    dfCorrLst = [pd.DataFrame(index=siteNoLst, columns=usgs.varC)
                 for x in range(2)]
    dfRmseLst = [pd.DataFrame(index=siteNoLst, columns=usgs.varC)
                 for x in range(2)]

    for siteNo in siteNoLst:
        outFolder = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-F')
        saveFile = os.path.join(outFolder, trainSet, siteNo)
        dfP = pd.read_csv(saveFile, index_col=None)
        # a bug - did not save dates
        startDate = pd.datetime(1979, 1, 1)
        endDate = pd.datetime(2020, 1, 1)
        ctR = pd.date_range(startDate, endDate)
        dfP.index = ctR
        dfP.index.name = 'date'
        dfY = pd.DataFrame({'date': ctR}).set_index('date')
        dfC, dfCF = usgs.readSample(siteNo, usgs.varC, flag=2)
        dfC[dfCF != 0] = np.nan
        dfY = dfY.join(dfC)
        yr = dfY.index.year.values
        indLst = [np.where(yr % 2 == x)[0] for x in [0, 1]]
        for code in usgs.varC:
            for k in range(2):
                ind = indLst[k]
                corr = dfY.iloc[ind][code].corr(dfP.iloc[ind][code])
                rmse = np.sqrt(
                    np.sum((dfY.iloc[ind][code]-dfP.iloc[ind][code])**2))
                dfCorrLst[k].loc[siteNo][code] = corr
                dfRmseLst[k].loc[siteNo][code] = corr
    for k in range(2):
        if k == 0:
            testSet = 'Yeven'
        else:
            testSet = 'Yodd'
        dfCorrLst[k].to_csv(os.path.join(
            outFolder, '{}-{}-corr'.format(trainSet, testSet)))
        dfRmseLst[k].to_csv(os.path.join(
            outFolder, '{}-{}-rmse'.format(trainSet, testSet)))

siteNo = '01013500'
code = '00681'
# fig, ax = plt.subplots(1, 1)
# ax.plot(dfC[code], '*r')
# ax.plot(dfP[code], '-b')
# fig.show()
