
import matplotlib
import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'rbWN5'
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
dictLSTMLst = list()
# LSTM
labelLst = ['Q_C', 'QFP_C', 'FP_QC']
for label in labelLst:
    dictLSTM = dict()
    trainSet = 'comb-B10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = basins.loadSeq(outName, siteNo)
        dictLSTM[siteNo] = df
    dictLSTMLst.append(dictLSTM)
# WRTDS
dictWRTDS = dict()
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'B10', 'output')
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    saveFile = os.path.join(dirWRTDS, siteNo)
    df = pd.read_csv(saveFile, index_col=None).set_index('date')
    # df = utils.time.datePdf(df)
    dictWRTDS[siteNo] = df
# Observation
dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(siteNo, varLst=codeLst, freq='W')
    dictObs[siteNo] = df

# calculate correlation
tt = np.datetime64('2010-01-01')
ind1 = np.where(df.index.values < tt)[0]
ind2 = np.where(df.index.values >= tt)[0]
dictLSTM = dictLSTMLst[1]
dictLSTM2 = dictLSTMLst[0]
corrMat = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
rmseMat = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v1 = dictLSTM[siteNo][code].iloc[ind2].values
        v2 = dictWRTDS[siteNo][code].iloc[ind2].values
        v0 = dictObs[siteNo][code].iloc[ind2].values
        [v1, v2, v0], ind = utils.rmNan([v1, v2, v0])
        rmse1, corr1 = utils.stat.calErr(v1, v0, rmExt=True)
        rmse2, corr2 = utils.stat.calErr(v2, v0, rmExt=True)
        rmse3, corr3 = utils.stat.calErr(v1, v2, rmExt=True)
        corrMat[indS, ic, 0] = corr1
        corrMat[indS, ic, 1] = corr2
        corrMat[indS, ic, 2] = corr3
        rmseMat[indS, ic, 0] = rmse1
        rmseMat[indS, ic, 1] = rmse2
        rmseMat[indS, ic, 2] = rmse3

# plot ts
code = '00915'
iCode = codeLst.index(code)
indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
siteNoLstCode = dictSite[code]
matMap = corrMat[indS, iCode, 0]-corrMat[indS, iCode, 1]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstCode)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
shortName = usgs.codePdf.loc[code]['shortName']


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(12, 4))
    axplot.mapPoint(axM, lat, lon, matMap, vRange=[-0.3, 0.3], s=16)
    axM.set_title('testing corr LSTM - corr WRTDS')
    figP = plt.figure(figsize=[16, 6])
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLstCode[iP]
    outName1 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QTFP_C', trainSet)
    dfL1 = basins.loadSeq(outName1, siteNo)
    dfW = pd.read_csv(os.path.join(dirWRTDS, siteNo),
                      index_col=None).set_index('date')
    dfO = waterQuality.readSiteTS(siteNo, codeLst+['00060'], freq='W')
    t = dfO.index
    # ts
    tBar = np.datetime64('2010-01-01')
    sd = np.datetime64('1980-01-01')
    legLst = ['LSTM', 'WRTDS', 'Obs']
    axplot.plotTS(axP, t, [dfL1[code], dfW[code], dfO[code]],
                  tBar=tBar, sd=sd, styLst='--*', cLst='rbk', legLst=legLst)
    corrL = corrMat[indS[iP], iCode, 0]
    corrW = corrMat[indS[iP], iCode, 1]
    axP.set_title('{} site {}; LSTM corr={:.2f} WRTDS corr={:.2f}'.format(
        shortName, siteNo, corrL, corrW))

    # axplot.titleInner(
    #     axP, 'siteNo {} {:.2f} {:.2f}'.format(siteNo, corrL, corrW))
    axP.legend()


matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

importlib.reload(axplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)

# 121 plot

# tempLst = ['09163500', '05465500', '02175000', '09171100']
tempLst = ['10343500', '05465500', '02175000', '09171100']

code = '00915'
fig, ax = plt.subplots(1, 1, figsize=[8, 6])
ic = codeLst.index(code)
x = corrMat[:, ic, 0]
y = corrMat[:, ic, 1]
c = corrMat[:, ic, 2]
out = axplot.scatter121(ax, x, y, c)
# for siteNo in tempLst:
#     indS = siteNoLst.index(siteNo)
#     circle = plt.Circle([x[indS], y[indS]], 0.05,
#                         color='black', fill=False)
#     ax.add_patch(circle)
ax.set_xlabel('LSTM correlation')
ax.set_ylabel('WRTDS correlation')
ax.set_title('Performance on Calcium')
fig.colorbar(out, ax=ax)
fig.show()

figM, axM = plt.subplots(1, 2, figsize=(12, 4))
indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
siteNoLstCode = dictSite[code]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstCode)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
axplot.mapPoint(axM[0], lat, lon, corrMat[indS, iCode, 0], vRange=[0, 1], s=16)
axplot.mapPoint(axM[1], lat, lon, corrMat[indS, iCode, 1], vRange=[0, 1], s=16)
axM[0].set_title('LSTM correlation of Ca')
axM[1].set_title('WRTDS correlation of Ca')
for siteNo in tempLst:
    for ax in axM:
        ind = siteNoLstCode.index(siteNo)
        circle = plt.Circle([lon[ind], lat[ind]], 1,
                            color='black', fill=False)
        ax.add_patch(circle)
figM.show()


for siteNo in tempLst:
    ind = siteNoLst.index(siteNo)
    figP, axP = plt.subplots(1, 1, figsize=(16, 4))
    outName1 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QTFP_C', trainSet)
    dfL1 = basins.loadSeq(outName1, siteNo)
    dfW = pd.read_csv(os.path.join(dirWRTDS, siteNo),
                      index_col=None).set_index('date')
    dfO = waterQuality.readSiteTS(siteNo, codeLst+['00060'], freq='W')
    t = dfO.index
    # ts
    tBar = np.datetime64('2010-01-01')
    sd = np.datetime64('1980-01-01')
    legLst = ['LSTM', 'WRTDS', 'Obs']
    axplot.plotTS(axP, t, [dfL1[code], dfW[code], dfO[code]],
                  tBar=tBar, sd=sd, styLst='--*', cLst='rbk', legLst=legLst)
    corrL = corrMat[ind, iCode, 0]
    corrW = corrMat[ind, iCode, 1]
    axP.set_title('{} site {}; LSTM corr={:.2f} WRTDS corr={:.2f}'.format(
        shortName, siteNo, corrL, corrW))
    axP.legend()
    figP.show()


np.where((x > 0.95) & (y > 0.95))[0]
siteNoLst[np.where((x > 0.95) & (y > 0.95))[0][0]]
# 09095500

np.where((x > 0.6) & (y < 0.4))[0][0]
siteNoLst[np.where((x > 0.6) & (y < 0.4))[0][0]]
# 05465500

np.where((x < 0.4) & (y > 0.6))[0][0]
siteNoLst[np.where((x < 0.4) & (y > 0.6))[0][0]]
# 02175000'

np.where((x < 0.4) & (y < 0.4) & (c > 0.6))[0]
siteNoLst[np.where((x < 0.4) & (y < 0.4) & (c > 0.6))[0][0]]
# 09171100'


np.where((x < 0.4) & (y < 0.4) & (c < 0.4))[0]
siteNoLst[np.where((x < 0.4) & (y < 0.4) & (c < 0.4))[0][0]]
# 05014300'
