
from mpl_toolkits import basemap
import pandas as pd
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.app.waterQuality import WRTDS
import statsmodels.api as sm
import scipy
from hydroDL.app.waterQuality import cqType
import importlib
import time

# load data
dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)
siteNoLst = DF.siteNoLst
codeLst = DF.varC
ns = len(siteNoLst)
nc = len(codeLst)

# load pars
filePar = os.path.join(kPath.dirWQ, 'modelStat', 'typeCQ', dataName+'.npz')
npz = np.load(filePar)
matA = npz['matA']
matB = npz['matB']
matP = npz['matP']

# get types
importlib.reload(axplot)
importlib.reload(cqType)
tp = cqType.par2type(matB, matP)

# plot map
saveDir = os.path.join(kPath.dirWQ, 'modelStat',
                       'typeCQ', dataName, 'bMaps')
if not os.path.exists(saveDir):
    os.mkdir(saveDir)
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values

for code in codeLst:

    indC = codeLst.index(code)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axplot.mapPoint(axes[0], lat, lon, matB[:, indC, 0], centerZero=True)
    axplot.mapPoint(axes[1], lat, lon, matB[:, indC, 1], centerZero=True)
    title = '{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
    axes[0].set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(saveDir, code))
    fig.show()
