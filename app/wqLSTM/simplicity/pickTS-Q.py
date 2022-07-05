import calendar
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import sklearn.tree
import matplotlib.gridspec as gridspec


# investigate correlation between simlicity and basin attributes.
# remove carbon - less obs, high corr

codeLst = usgs.varC

DF = dbBasin.DataFrameBasin('G200')
# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)

matRm = count < 200

# load linear/seasonal
dirP = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\{}\param'
labLst = ['Q', 'S', 'QS']
dictS = dict()
for lab in labLst:
    dirS = dirP.format(lab)
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirS, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    dictS[lab] = matLR

matQ = dictS['Q']
matS = dictS['S']
matQS = dictS['QS']

# global variables

td = pd.to_datetime(DF.t).dayofyear
lat, lon = DF.getGeo()

# linearity
code = '00915'
indC = codeLst.index(code)
indS = np.where(~matRm[:, indC])[0]


def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(
        figM, gsM[0, 0], lat[indS], lon[indS], matQ[indS, indC])
    axM.set_title('{} {}'.format(usgs.codePdf.loc[code]['shortName'], code))
    figP = plt.figure(figsize=(15, 3))
    gsP = gridspec.GridSpec(1, 3)
    ax = figP.add_subplot(gsP[0, 0:2])
    axS = figP.add_subplot(gsP[0, 2])
    axT = ax.twinx()    
    axPLst = [ax, axT, axS]
    axP = np.array(axPLst)
    return figM, axM, figP, axP, lon[indS], lat[indS]


def funcP(iP, axP):
    print(iP)
    iS = indS[iP]
    [ax, axT, axS] = axP
    q = DF.q[:, iS, 1]
    c = DF.c[:, iS, indC]
    axplot.plotTS(axT, DF.t, q, cLst='b', lineW=[0.1])
    axplot.plotTS(ax, DF.t, c, cLst='r')
    codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
    ax.set_title('{} {} linearity = {:.2f}, seasonality = {:.2f}'.format(
        DF.siteNoLst[iS], codeStr, matQ[iS, indC], matS[iS, indC]))
    sc = axS.scatter(np.log(q), c, c=td, cmap='hsv_r', vmin=0, vmax=365)


figplot.clickMap(funcM, funcP)


# list of month first day
dLst = list()
sLst = list()
for m in range(12):
    s = calendar.month_abbr[m+1]+'-01'
    d = pd.to_datetime('2021-{}'.format(s)).dayofyear
    dLst.append(d)
    sLst.append(s)

d = np.arange(0, 366)
z = np.arange(40, 70)
v = d * np.ones((30, 365))
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
ax.pcolormesh(d*np.pi/365*2, z, v, cmap='hsv_r')
ax.set_yticks([])
ax.set_xticks(np.array(dLst)*np.pi/365*2)
ax.set_xticklabels(sLst)
ax.set_rorigin(-2.5)
fig.show()
