
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

import warnings

DF = dbBasin.DataFrameBasin('G200')
codeLst = DF.varC
nc = len(codeLst)
ns = len(DF.siteNoLst)
matCount = np.sum(np.isnan(DF.c), axis=0)

# remove high/low values
data = DF.c.copy()
p = 5
v1 = np.nanpercentile(data, p, axis=0)
v2 = np.nanpercentile(data, 100-p, axis=0)
b1 = data > v2
b2 = data < v1
data[b1] = np.nan
data[b2] = np.nan

matMean = np.nanmean(data, axis=0)
matStd = np.nanstd(data, axis=0)
n = 200
matMean[matCount < n] = np.nan
matStd[matCount < n] = np.nan


nfy, nfx = [4, 5]
fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, nfy, nfx)
    kc = DF.varC.index(code)
    # axplot.plot121(axes[j, i], matMean[:, kc], matStd[:, kc])
    axes[j, i].plot(matMean[:, kc], matStd[:, kc], '*')
    title = '{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
    axplot.titleInner(axes[j, i],title)    
fig.show()
