from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# ts map of single dataset, label and code
freq = 'W'
dirRoot1 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly')
dirRoot2 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly_rmq')

code = '00955'
dfRes1 = pd.read_csv(os.path.join(dirRoot1, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')
dfRes2 = pd.read_csv(os.path.join(dirRoot2, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')
dfGeo = gageII.readData(siteNoLst=dfRes1.index.tolist())
dfGeo = gageII.updateCode(dfGeo)

# plot map
nS = 100
dfR1 = dfRes1[dfRes1['count'] > nS]
siteNoLst = dfR1.index.tolist()
dfR2 = dfRes2.loc[siteNoLst]
dfG = dfGeo.loc[siteNoLst]


# calculate freatures from data
colLst = ['mean', 'std']
dfStat = pd.DataFrame(index=siteNoLst, columns=colLst, dtype=float)
ul = 120
for siteNo in siteNoLst:
    dfO = waterQuality.readSiteTS(siteNo, [code], freq=freq)
    dfO.at[dfO[code] > ul, code] = np.nan
    data = dfO[code].dropna().values
    dfStat.at[siteNo, 'mean'] = np.mean(data)
    dfStat.at[siteNo, 'std'] = np.std(data)

# plot vs
fig, ax = plt.subplots(1, 1)
ax.plot(dfStat['mean'].values, dfR1['corr'].values, '*')
fig.show()

fig, ax = plt.subplots(1, 1)
mat1 = dfStat['mean'].values
# mat2 = dfStat['std'].values
mat2 = dfG['BFI_AVE'].values
matR = dfR1['corr'].values
im = ax.scatter(mat1, mat2, c=matR, cmap='jet')
fig.colorbar(im, ax=ax)
fig.show()

dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def funcMap():
    figM, axM = plt.subplots(1, 2, figsize=(12, 6))
    axplot.mapPoint(axM[0], lat, lon, matR, s=16)
    axplot.mapPoint(axM[1], lat, lon, mat1, s=16)
    shortName = usgs.codePdf.loc[code]['shortName']
    figP, axP = plt.subplots(2, 1, figsize=(16, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfO = waterQuality.readSiteTS(siteNo, ['00060', code], freq=freq)
    file1 = os.path.join(dirRoot1, 'output', siteNo)
    dfP = pd.read_csv(file1, index_col='date')
    t = dfO.index.values
    axplot.plotTS(axP[0], t, dfO['00060'].values, styLst='-*', cLst='bgr')
    axplot.plotTS(axP[1], t, dfP[code].values, styLst='-', cLst='r')
    axplot.plotTS(axP[1], t, dfO[code].values, styLst='*', cLst='b')
    axP[0].set_title(siteNo)


figM, figP = figplot.clickMap(funcMap, funcPoint)
