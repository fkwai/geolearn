import calendar
from matplotlib import style
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

xMat = np.ndarray([len(indS), 3])
yMat = np.ndarray([len(indS), 3])
xMat[:, 0] = matQ[indS, indC]
yMat[:, 0] = matS[indS, indC]
xMat[:, 1] = lon[indS]
yMat[:, 1] = lat[indS]
xMat[:, 2] = lon[indS]
yMat[:, 2] = lat[indS]


def funcM():
    figM = plt.figure(figsize=(16, 4))
    gsM = gridspec.GridSpec(1, 5)
    labelLst = ['scatter', 'mapL', 'mapS']
    axS = figM.add_subplot(gsM[0, :1])
    axS.set_label(labelLst[0])
    cs = axplot.scatter121(axS, xMat[:, 0], yMat[:, 0],  matQS[indS, indC])
    axS.set_xlabel('linearity')
    axS.set_ylabel('seasonality')
    plt.colorbar(cs, orientation='vertical',label='simplicity')
    axM1 = mapplot.mapPoint(
        figM, gsM[0, 1:3], lat[indS], lon[indS], matQ[indS, indC])
    axM1.set_label(labelLst[1])
    axM1.set_title('linearity')
    axM2 = mapplot.mapPoint(
        figM, gsM[0, 3:], lat[indS], lon[indS], matS[indS, indC])
    axM2.set_title('seasonality')
    axM2.set_label(labelLst[2])
    axM = np.array([axS, axM1, axM2])
    figP = plt.figure(figsize=(15, 3))
    gsP = gridspec.GridSpec(1, 3)
    ax = figP.add_subplot(gsP[0, 0:2])
    axS = figP.add_subplot(gsP[0, 2])
    axT = ax.twinx()
    axPLst = [ax, axT, axS]
    axP = np.array(axPLst)
    return figM, axM, figP, axP, xMat, yMat, labelLst


def funcP(axP, iP, iM):
    print(iP)
    iS = indS[iP]
    [ax, axT, axS] = axP
    q = DF.q[:, iS, 1]
    c = DF.c[:, iS, indC]
    axplot.plotTS(axT, DF.t, q, cLst='b', styLst=['-'], lineW=[0.1])
    axplot.plotTS(ax, DF.t, c, cLst='r')
    codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
    ax.set_title('{} {} sim={:.2f}; lin = {:.2f}; sea = {:.2f}'.format(
        DF.siteNoLst[iS], codeStr, matQS[iS, indC], matQ[iS, indC], matS[iS, indC]))
    sc = axS.scatter(np.log(q), c, c=td, cmap='hsv_r', vmin=0, vmax=365)


figM, figP = figplot.clickMulti(funcM, funcP)


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
