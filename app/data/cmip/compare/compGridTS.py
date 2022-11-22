
import hydroDL.utils.ts
from scipy import interpolate
import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.data.cmip.io
import hydroDL.data.gridMET.io
import importlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot

importlib.reload(hydroDL.data)

# set dates
d1 = np.datetime64('2013-01-01')
d2 = np.datetime64('2015-01-01')
d3 = np.datetime64('2017-01-01')
y1, y2, y3 = [d.astype(object).year for d in [d1, d2, d3]]
latR = [25, 50]
lonR = [-125, -65]
varG = 'pr'
varC = 'pr'
func = 'nansum'

varG = 'tmmx'
varC = 'tasmax'
func = 'nanmean'


# read gridMet
gridF1, (latG, lonG, tG1) = hydroDL.data.gridMET.io.read(
    varG, y1, y2, dtype='float32')
gridF2, (latG, lonG, tG2) = hydroDL.data.gridMET.io.read(
    varG, y2, y3, dtype='float32')


# read CMIP6
df = hydroDL.data.cmip.io.walkFile()
modelName = 'MPI-ESM1-2-XR'
data1, latC1, lonC1, tC1 = hydroDL.data.cmip.io.readCMIP(
    dfFile=df, var=varC, exp='hist-1950', latR=latR, lonR=lonR,
    sd=d1, ed=d2, model=modelName)

data2, latC2, lonC2, tC2 = hydroDL.data.cmip.io.readCMIP(
    dfFile=df, var=varC, exp='highres-future', latR=latR, lonR=lonR,
    sd=d2, ed=d3, model=modelName)

# check consistance
latC = latC1 if np.array_equal(latC1, latC2) else Exception('latC')
lonC = lonC1 if np.array_equal(lonC1, lonC2) else Exception('lonC')
t1 = tG1 if np.array_equal(tG1, tC1) else Exception('t1')
t2 = tG2 if np.array_equal(tG2, tC2) else Exception('t2')


# interpolate
latM, lonM = np.meshgrid(latC, lonC, indexing='ij')
interp = interpolate.RegularGridInterpolator(
    (latG[::-1], lonG), gridF1[::-1, :, :], bounds_error=False, fill_value=np.nan)
grid1 = interp((latM, lonM))
interp = interpolate.RegularGridInterpolator(
    (latG[::-1], lonG), gridF2[::-1, :, :], bounds_error=False, fill_value=np.nan)
grid2 = interp((latM, lonM))


fig = plt.figure()
gs = gridspec.GridSpec(1, 2)
ax = mapplot.mapGrid(fig, gs[0, 0], latG, lonG,
                     np.mean(gridF1, axis=-1), extent=None)
ax = mapplot.mapGrid(fig, gs[0, 1], latC, lonC,
                     np.mean(grid1, axis=-1), extent=None)
fig.show()


# calculate corr
rF1 = hydroDL.utils.stat.gridCorrT(grid1, data1)
rF2 = hydroDL.utils.stat.gridCorrT(grid2, data2)

gridM1, _ = hydroDL.utils.ts.data2Monthly(grid1, t1, func=func)
dataM1, _ = hydroDL.utils.ts.data2Monthly(data1, t1, func=func)
gridM2, _ = hydroDL.utils.ts.data2Monthly(grid2, t2, func=func)
dataM2, _ = hydroDL.utils.ts.data2Monthly(data2, t2, func=func)
rM1 = hydroDL.utils.stat.gridCorrT(gridM1, dataM1)
rM2 = hydroDL.utils.stat.gridCorrT(gridM2, dataM2)

gridA1, _ = hydroDL.utils.ts.data2Climate(grid1, t1, func=func)
dataA1, _ = hydroDL.utils.ts.data2Climate(data1, t1, func=func)
gridA2, _ = hydroDL.utils.ts.data2Climate(grid2, t2, func=func)
dataA2, _ = hydroDL.utils.ts.data2Climate(data2, t2, func=func)
rA1 = hydroDL.utils.stat.gridCorrT(gridA1, dataA1)
rA2 = hydroDL.utils.stat.gridCorrT(gridA2, dataA2)


fig = plt.figure(figsize=(16, 4))
vR = [0.5, 1]
gs = gridspec.GridSpec(1, 2)
ax = mapplot.mapGrid(fig, gs[0, 0], latC, lonC,
                     rF1, vRange=vR, cbOri='horizontal')
ax = mapplot.mapGrid(fig, gs[0, 1], latC, lonC,
                     rF2, vRange=vR, cbOri='horizontal')
fig.suptitle('r of daily comparison, {}'.format(varC))
fig.show()


fig = plt.figure(figsize=(16, 4))
vR = [0.5, 1]
gs = gridspec.GridSpec(1, 2)
ax = mapplot.mapGrid(fig, gs[0, 0], latC, lonC,
                     rM1, vRange=vR, cbOri='horizontal')
ax = mapplot.mapGrid(fig, gs[0, 1], latC, lonC,
                     rM2, vRange=vR, cbOri='horizontal')
fig.suptitle('r of monthly comparison, {}'.format(varC))
fig.show()


fig = plt.figure(figsize=(16, 4))
vR = [0.5, 1]
gs = gridspec.GridSpec(1, 2)
ax = mapplot.mapGrid(fig, gs[0, 0], latC, lonC,
                     rA1, vRange=vR, cbOri='horizontal')
ax = mapplot.mapGrid(fig, gs[0, 1], latC, lonC,
                     rA2, vRange=vR, cbOri='horizontal')
fig.suptitle('r of climate comparison, {}'.format(varC))
fig.show()


# tsMap
dataPlot = [[gridF1, grid1, data1], [gridF2, grid2, data2]]


def funcM():
    vR = None
    figM = plt.figure(figsize=(15, 5))
    gsM = gridspec.GridSpec(2, 3)
    axMLst, xLst, yLst = (list(), list(), list())
    lonPlot = [[lonG, lonC, lonC], [lonG, lonC, lonC]]
    latPlot = [[latG, latC, latC], [latG, latC, latC]]
    strY = ['{}-{}'.format(y1, y2), '{}-{}'.format(y2, y3)]
    strX = ['gridMET', 'gridMET Coarse', modelName]
    for j in range(2):
        for i in range(3):
            strTitle = 'avg {} {} {}'.format(varC, strY[j], strX[i])
            data = dataPlot[j][i]
            lat = latPlot[j][i]
            lon = lonPlot[j][i]
            axM = mapplot.mapGrid(
                figM, gsM[j, i], lat, lon, np.mean(data, axis=-1), vRange=vR)
            axM.set_title(strTitle)
            axMLst.append(axM)
            xLst.append(lon)
            yLst.append(lat)
    gsP = gridspec.GridSpec(2, 2)
    figP = plt.figure(figsize=[15, 5])
    axP1 = figP.add_subplot(gsP[0, :])
    axP2 = figP.add_subplot(gsP[1, 0])
    axP3 = figP.add_subplot(gsP[1, 1])
    axP = np.array([axP1, axP2, axP3])
    return figM, axMLst, figP, axP, xLst, yLst


def funcP(axP, yy, xx):
    jG, iG = figplot.findPoint(xx, yy, lonG, latG)
    jC, iC = figplot.findPoint(xx, yy, lonC, latC)
    labLst = ['gridMET', 'gridMET-coarse', modelName]
    cLst = 'kbr'
    tLst = [t1, t2]
    for j, t in enumerate(tLst):
        for i, (lab, c, jj, ii) in enumerate(
                zip(labLst, cLst, [jG, jC, jC], [iG, iC, iC])):
            temp = dataPlot[j][i][jj, ii, :]
            tempM, tM = hydroDL.utils.ts.data2Monthly(temp, t, func=func)
            tempA, tA = hydroDL.utils.ts.data2Climate(temp, t, func=func)
            if j == 0:
                axP[0].plot(t, temp, color=c, label=lab)
                axP[2].plot(tA, tempA, color=c)
            else:
                axP[0].plot(t, temp, color=c)
                axP[2].plot(tA+365, tempA, color=c)
            axP[1].plot(tM, tempM, color=c)
    axP[0].axvline(d2, color='k')
    axP[1].axvline(d2, color='k')
    axP[2].axvline(365, color='k')
    axP[0].legend()


figplot.clickMultiMap(funcM, funcP)
