import os
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, ntn
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
crdNTN = ntn.loadSite()
ntnIdLst = crdNTN.index.tolist()
freq = 'W'
# read all ntn
dictNTN = dict()
for k, ntnId in enumerate(ntnIdLst):
    print(k, ntnId)
    tab = ntn.readSite(ntnId, freq)
    dictNTN[ntnId] = tab

varLst = ntn.varLst[2:-1]
dfCount = pd.DataFrame(index=crdNTN.index, columns=varLst, dtype=float)
dfMean = dfCount.copy()
dfStd = dfCount.copy()

nc = len(varLst)
ns = len(crdNTN)
for k, ntnId in enumerate(ntnIdLst):
    print(k, ntnId)
    tab = dictNTN[ntnId]
    dfCount.loc[ntnId] = len(tab)-tab.isna().sum()
    dfMean.loc[ntnId] = tab.mean()
    dfStd.loc[ntnId] = tab.std()

lat = crdNTN['latitude']
lon = crdNTN['longitude']
figM = plt.figure()
gsM = gridspec.GridSpec(4, 2)
for k, var in enumerate(varLst):
    iy, ix = utils.index2d(k, 4, 2)
    axM = mapplot.mapPoint(
        figM, gsM[iy, ix], lat, lon,
        dfStd[var], s=16, cb=True)
    axM.set_title('{}'.format(var))

figM.show()


def funcM():
    figM = plt.figure()
    gsM = gridspec.GridSpec(4, 2)
    axM = list()
    for k, var in enumerate(varLst):
        iy, ix = utils.index2d(k, 4, 2)
        axM = mapplot.mapPoint(
            figM, gsM[iy, ix], lat, lon,
            dfCount[var], s=16, cb=False)
        axM.set_title('{}'.format(var))
    figP, axP = plt.subplots(nc, 1, sharex=True)
    figP.subplots_adjust(hspace=0.1)

    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = dfCount.index[iP]
    df = dictNTN[siteNo]
    for k, var in enumerate(varLst):
        axP[k].plot(df[var], '*')
        titleStr = '{} {:.0f}'.format(var, dfCount.at[siteNo, var])
        axplot.titleInner(axP[k], titleStr)


figM, figP = figplot.clickMap(funcM, funcP)
