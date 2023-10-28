import pandas as pd
from hydroDL.data import usgs
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath
import os
from hydroDL.app.waterQuality import cqType
import importlib

import matplotlib.gridspec as gridspec


tabCrd=pd.read_csv(os.path.join(kPath.dirUsgs,  'index', 'siteGageII.csv'),
                   dtype={'siteNo':str}).set_index('siteNo')
tabCount=pd.read_csv(os.path.join(kPath.dirUsgs,  'index', 'sampleCount_a79_v20.csv'),
                   dtype={'siteNo':str}).set_index('siteNo')

code='00945'

# plot cdf
fig, ax = plt.subplots(1, 1)
count=tabCount[code].values
ax.plot(np.sort(count),'*')
fig.show()

the=100
tabSel=tabCount.loc[tabCount[code]>the,code]

siteNoLst=tabSel.index.tolist()
lat=tabCrd.loc[siteNoLst]['lat'].values
lon=tabCrd.loc[siteNoLst]['lon'].values
count=tabSel.values


figM, axM = plt.subplots(1, 1, figsize=(8, 6))
gs = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gs[0, 0], lat, lon, count)
figM.show()

def funcM():
    figM, axM = plt.subplots(1, 1, figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(fig, gs[0, 0], lat, lon, count)
    for k, code in enumerate(codePlot):
        indC = codeLst.index(code)
        ax2 = mapplot.mapPoint(
                fig, gs[0, 1], lat2, lon2, v2, vRange=vRange, extent=extent
            )
        axplot.mapPointClass(axM[k], lat, lon, tp[:, indC],
                             vLst=vLst, mLst=mLst, cLst=cLst, labLst=labLst)
        title = '{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
        axM[k].set_title(title)
    figP, axP = plt.subplots(nM, 1, figsize=(8, 6))
    axP = np.array([axP]) if nM == 1 else axP
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    siteNo = siteNoLst[iP]
    for k, code in enumerate(codePlot):
        indC = codeLst.index(code)
        Q = DF.q[:, iP, 1]
        C = DF.c[:, iP, indC]
        a = matA[iP, indC, :]
        b = matB[iP, indC, :]
        p = matP[iP, indC, :]
        cqType.plotCQ(axP[k], Q, C, a, b, p)
        title = '{} {} {}'.format(
            siteNo, usgs.codePdf.loc[code]['shortName'], code)
        axP[k].set_title(title)


importlib.reload(figplot)
figM, figP = figplot.clickMap(funcM, funcP)
