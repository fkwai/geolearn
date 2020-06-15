import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, wqLinear, wqRela
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# test
outName = 'Silica64-00955-Y8090-opt1'

wqData = waterQuality.DataModelWQ('Silica64')
code = '00955'
trainset = 'Y8090'
testset = 'Y0010'
master = basins.loadMaster(outName)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)

siteNo = siteNoLst[0]
dfPred, dfObs = basins.loadSeq(outName, siteNo)

fig, axes = plt.subplots(2, 1)
axes[0].plot(np.log(dfPred['00060']), dfPred['00955'], '*')
axes[1].plot(np.log(dfObs['00060']), dfObs['00955'], '*')
fig.show()

ceqMat1 = np.full([len(siteNoLst), 2], np.nan)
dwMat1 = np.full([len(siteNoLst), 2], np.nan)
ceqMat2 = np.full([len(siteNoLst), 2], np.nan)
dwMat2 = np.full([len(siteNoLst), 2], np.nan)
sd = np.datetime64('1980-01-01')
tBar = np.datetime64('2000-01-01')
importlib.reload(wqRela)
for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    dfP, dfO = basins.loadSeq(outName, siteNo)
    dfP1 = dfP[(dfP.index < tBar) & (dfP.index > sd)]
    dfO1 = dfO[(dfO.index < tBar) & (dfO.index > sd)]
    dfP2 = dfP[dfP.index >= tBar]
    dfO2 = dfO[dfO.index >= tBar]
    try:
        cO1, dO1, _ = wqRela.kateModel(np.log(dfO1['00060']+1), dfO1['00955'])
        cP1, dP1, _ = wqRela.kateModel(np.log(dfP1['00060']+1), dfP1['00955'])
        cO2, dO2, _ = wqRela.kateModel(np.log(dfO2['00060']+1), dfO2['00955'])
        cP2, dP2, _ = wqRela.kateModel(np.log(dfP2['00060']+1), dfP2['00955'])
        ceqMat1[k, :] = [cO1, cP1]
        dwMat1[k, :] = [dO1, dP1]
        ceqMat2[k, :] = [cO2, cP2]
        dwMat2[k, :] = [dO2, dP2]
    except:
        pass

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(ceqMat1[:, 0], ceqMat1[:, 1], '*')
axes[0, 1].plot(ceqMat2[:, 0], ceqMat2[:, 1], '*')
axes[1, 0].plot(dwMat1[:, 0], dwMat1[:, 1], '*')
axes[1, 1].plot(dwMat2[:, 0], dwMat2[:, 1], '*')
fig.show()

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(ceqMat1[:, 0], ceqMat1[:, 1], '*')
axes[0, 1].plot(ceqMat2[:, 0], ceqMat2[:, 1], '*')
axes[1, 0].plot(dwMat1[:, 0], dwMat1[:, 1], '*')
axes[1, 1].plot(dwMat2[:, 0], dwMat2[:, 1], '*')
for i in range(2):
    for j in range(2):
        axes[i, j].set_xlim(0, 40)
        axes[i, j].set_ylim(0, 40)
        axes[i, j].plot([0, 40], [0, 40], '-k')
fig.show()

# time series map
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(2, 1, figsize=(8, 6))
    # shortName = codePdf.loc[code]['shortName']
    axplot.mapPoint(axM[0], lat, lon, dwMat2[:,0], s=12)
    axplot.mapPoint(axM[1], lat, lon, dwMat2[:,1], s=12)
    # axM[k].set_title(modLst[k])
    figP, axP = plt.subplots(2, 2, figsize=(8, 6))    
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfP, dfO = basins.loadSeq(outName, siteNo)
    dfP1 = dfP[(dfP.index < tBar) & (dfP.index > sd)]
    dfO1 = dfO[(dfO.index < tBar) & (dfO.index > sd)]
    dfP2 = dfP[dfP.index >= tBar]
    dfO2 = dfO[dfO.index >= tBar]
    axP[0, 0].plot(np.log(dfO1['00060']), dfO1['00955'], '*')
    axP[0, 0].set_title('{} B2000 observation'.format(siteNo))
    axP[0, 1].plot(np.log(dfO2['00060']), dfO2['00955'], '*')
    axP[0, 1].set_title('{} A2000 observation'.format(siteNo))
    axP[1, 0].plot(np.log(dfP1['00060']), dfP1['00955'], '*')
    axP[1, 0].set_title('{} B2000 prediction'.format(siteNo))
    axP[1, 1].plot(np.log(dfP2['00060']), dfP2['00955'], '*')
    axP[1, 1].set_title('{} A2000 prediction'.format(siteNo))


importlib.reload(figplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)

for ax in figP.axes:
    ax.set_xlim(figP.axes[-1].get_xlim())
    ax.set_ylim(figP.axes[-1].get_ylim())
figP.canvas.draw()