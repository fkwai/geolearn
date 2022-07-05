import string
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

dirOut = r'C:\Users\geofk\work\waterQuality\paper\G200\simplicity'

dictPlot = {
    '00945': ['09371492',
              '09066510',
              '401733105392404',
              '402114105350101',
              '06436198',
              '04231000'],
    '00930': ['11264500',
              '09066510',
              '402114105350101',
              '09095500',
              '09371492',
              '13092747']}

# map with circles
lat, lon = DF.getGeo()
td = pd.to_datetime(DF.t).dayofyear
code = '00930'
codeStr = usgs.codePdf.loc[code]['shortName']
indC = codeLst.index(code)
indS = np.where(~matRm[:, indC])[0]
td = pd.to_datetime(DF.t).dayofyear
siteNoPlot = dictPlot[code]
labPlot = string.ascii_uppercase[:len(siteNoPlot)]

outFolder = os.path.join(dirOut, 'exampleTS', 'QS')
if not os.path.exists(outFolder):
    os.makedirs(outFolder)
fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(2, 5)
axM1 = mapplot.mapPoint(
    fig, gs[0, :3], lat[indS], lon[indS], matQ[indS, indC])
axM1.set_title('linearity of {} {}'.format(codeStr, code))
axM2 = mapplot.mapPoint(
    fig, gs[1, :3], lat[indS], lon[indS], matS[indS, indC])
axM2.set_title('seasonality of {} {}'.format(codeStr, code))
for siteNo, lab in zip(siteNoPlot, labPlot):
    xLoc = lon[DF.siteNoLst.index(siteNo)]
    yLoc = lat[DF.siteNoLst.index(siteNo)]
    circle = plt.Circle([xLoc, yLoc], 1,
                        color='black', fill=False)
    axM1.add_patch(circle)
    axM1.text(xLoc+0.1, yLoc+0.1, lab, fontsize=16)
for siteNo, lab in zip(siteNoPlot, labPlot):
    xLoc = lon[DF.siteNoLst.index(siteNo)]
    yLoc = lat[DF.siteNoLst.index(siteNo)]
    circle = plt.Circle([xLoc, yLoc], 1,
                        color='black', fill=False)
    axM2.add_patch(circle)
    axM2.text(xLoc+0.1, yLoc+0.1, lab, fontsize=16)
axP = fig.add_subplot(gs[:, 3:])
axP.scatter(matQ[indS, indC], matS[indS, indC], c=matQS[indS, indC])
axP.set_xlabel('linearity of {} {}'.format(codeStr, code))
axP.set_ylabel('seasonality of {} {}'.format(codeStr, code))
axP.plot([0, 1], [0, 1], '-k')
axP.set_aspect(1)
for siteNo, lab in zip(siteNoPlot, labPlot):
    xLoc = matQ[DF.siteNoLst.index(siteNo), indC]
    yLoc = matS[DF.siteNoLst.index(siteNo), indC]
    circle = plt.Circle([xLoc, yLoc], 0.01,
                        color='black', fill=False)
    axP.add_patch(circle)
    axP.text(xLoc, yLoc, lab, fontsize=16)
fig.show()
fig.savefig(os.path.join(outFolder, 'map_{}'.format(code)))


# linearity
for siteNo, lab in zip(siteNoPlot, labPlot):
    iP = DF.siteNoLst.index(siteNo)
    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, 0:2])
    axS = fig.add_subplot(gs[0, 2])
    axT = ax.twinx()
    q = DF.q[:, iP, 1]
    c = DF.c[:, iP, indC]
    axT.invert_yaxis()
    axplot.plotTS(axT, DF.t, q, cLst='b', styLst='-',
                  lineW=[0.1], legLst=['q'])
    axplot.plotTS(ax, DF.t, c, cLst='r')
    codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
    ax.set_title('{} {} {} sim={:.2f}; lin = {:.2f}; sea = {:.2f}'.format(
        lab, DF.siteNoLst[iP], codeStr, matQS[iP, indC], matQ[iP, indC], matS[iP, indC]))
    sc = axS.scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)
    axS.set_xlabel('logQ', loc='right')
    axS.set_ylabel('C', loc='top')
    plt.tight_layout()

    fig.show()
    fig.savefig(os.path.join(
        outFolder, 'TS_{}_{}_{}'.format(lab, code, siteNo)))
