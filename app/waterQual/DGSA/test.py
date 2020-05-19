from hydroDL import kPath
from hydroDL.app import waterQuality, DGSA
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time

wqData = waterQuality.DataModelWQ('Silica64')

siteNoLst = wqData.siteNoLst
rMat = np.ndarray([len(siteNoLst), 2])
for k, siteNo in enumerate(siteNoLst):
    print(siteNo)
    dfObs = waterQuality.readSiteY(siteNo, ['00955'])
    rMat[k, :] = [dfObs.mean(), dfObs.std()]

dfG = gageII.readData(varLst=gageII.varLst, siteNoLst=siteNoLst)
dfG = gageII.updateCode(dfG)

pMat = dfG.values
dfS = DGSA.DGSA_light(
    pMat, rMat, ParametersNames=dfG.columns.tolist(), n_clsters=3)
# ax = dfS.sort_values(by=0).plot.barh()
# plt.show()

dfSP = dfS.sort_values(by=0)
fig, ax = plt.subplots(1, 1)
x = range(len(dfSP))
cLst = list()
for b in (dfSP[0] > 1).tolist():
    cLst.append('r') if b is True else cLst.append('b')
ax.barh(x, dfSP[0].values, color=cLst)
ax.set_yticks(x)
ax.set_yticklabels(dfSP.index.tolist())
plt.tight_layout()
fig.show()