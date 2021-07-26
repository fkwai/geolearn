
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
code = '00955'
indC = codeLst.index(code)
tpC = tp[:, indC]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
fig, ax = plt.subplots(1, 1)
vLst, cLst,  mLst, labLst = cqType.getPlotArg()
axplot.mapPointClass(ax, lat, lon, tp[:, indC], vLst=vLst, mLst=mLst,
                     cLst=cLst, labLst=labLst)
fig.show()

# CQ plot
indS = np.where(tpC == 4)[0][10]
fig, ax = plt.subplots(1, 1)
Q = DF.q[:, indS, 1]
C = DF.c[:, indS, indC]
a = matA[indS, indC, :]
b = matB[indS, indC, :]
p = matP[indS, indC, :]
cqType.plotCQ(ax, Q, C, a, b, p)
fig.show()

# tsmap
codePlot = ['00915', '00955']
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
vLst, cLst,  mLst, labLst = cqType.getPlotArg()


def funcM():
    nM = len(codePlot)
    figM, axM = plt.subplots(nM, 1, figsize=(8, 6))
    for k, code in enumerate(codePlot):
        indC = codeLst.index(code)
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
