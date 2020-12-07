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

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

df = pd.DataFrame(index=siteNoLst, columns=usgs.newC)
df.index.name = 'siteNo'
dfCorr = df.copy()
dfRmse = df.copy()

dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-D', 'All')
dirOut = os.path.join(dirWRTDS, 'output')
dirPar = os.path.join(dirWRTDS, 'params')

t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0))
    saveFile = os.path.join(dirOut, siteNo)
    dfP = pd.read_csv(saveFile, index_col=None).set_index('date')
    dfP.index = pd.to_datetime(dfP.index)
    dfC = waterQuality.readSiteTS(siteNo, varLst=usgs.newC)
    yr = dfC.index.year.values
    for code in usgs.newC:
        rmse, corr = utils.stat.calErr(
            dfP[code].values,  dfC[code].values)
        dfCorr.loc[siteNo][code] = corr
        dfRmse.loc[siteNo][code] = rmse

dfCorr.to_csv(os.path.join(dirWRTDS, 'corr'))
dfRmse.to_csv(os.path.join(dirWRTDS, 'rmse'))