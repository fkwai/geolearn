
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
                       'typeCQ', dataName, 'typeMaps')
if not os.path.exists(saveDir):
    os.mkdir(saveDir)
for code in codeLst:
    indC = codeLst.index(code)
    tpC = tp[:, indC]
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    vLst, cLst,  mLst, labLst = cqType.getPlotArg()
    axplot.mapPointClass(ax, lat, lon, tp[:, indC], vLst=vLst, mLst=mLst,
                         cLst=cLst, labLst=labLst)
    title = '{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(saveDir, code))
    fig.show()
