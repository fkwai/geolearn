
import matplotlib
# from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec
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

code = '00010'
siteNoLst = dictSite[code]
nSite = len(siteNoLst)
dataName = 'rbWN5'

# load all sequence
dictLSTMLst = list()
# LSTM
label = 'QTFP_C'
dictLSTM = dict()
trainSet = '{}-B10'.format('comb')
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = basins.loadSeq(outName, siteNo)
    dictLSTM[siteNo] = df

# WRTDS
dictWRTDS = dict()
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'Linear-W', 'B10Q', 'output')
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
    df = waterQuality.readSiteTS(siteNo, varLst=[code], freq='W')
    dictObs[siteNo] = df


# code = '00010'
# # calculate correlation
tt = np.datetime64('2010-01-01')
ind1 = np.where(df.index.values < tt)[0]
ind2 = np.where(df.index.values >= tt)[0]
corrLSTM = np.full([len(siteNoLst), 2], np.nan)
rmseLSTM = np.full([len(siteNoLst),  2], np.nan)
corrWRTDS = np.full([len(siteNoLst), 2], np.nan)
rmseWRTDS = np.full([len(siteNoLst), 2], np.nan)
for k, indT in enumerate([ind1, ind2]):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v0 = dictObs[siteNo][code].iloc[indT].values
        v1 = dictLSTM[siteNo][code].iloc[indT].values
        v2 = dictWRTDS[siteNo][code].iloc[indT].values
        v3 = dictObs[siteNo][code].iloc[indT].values
        [v0, v1, v2], ind = utils.rmNan([v0, v1, v2])
        rmse1, corr1 = utils.stat.calErr(v1, v0, rmExt=True)
        rmse2, corr2 = utils.stat.calErr(v2, v0, rmExt=True)
        corrLSTM[indS, k] = corr1
        corrWRTDS[indS, k] = corr2
        rmseLSTM[indS, k] = rmse1
        rmseWRTDS[indS, k] = rmse2

# box

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 12})
# # plot box
# labLst1 = [usgs.codePdf.loc[code]['shortName'] +
#            '\n'+code for code in codeLst]
label2 = ['train', 'test']
label1 = ['correlation', 'RMSE']
dataBox = list()
ic = 0
# dataBox = [[corrLSTM[:, 0], corrLSTM[:, 1],],
#            [rmseLSTM[:, 0], rmseLSTM[:, 1]]]
dataBox = [[corrLSTM[:, 0], corrLSTM[:, 1], corrWRTDS[:, 0], corrWRTDS[:, 1]],
           [rmseLSTM[:, 0], rmseLSTM[:, 1], rmseWRTDS[:, 0], rmseWRTDS[:, 1]]]
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='brgb',
                      label2=label2, label1=label1,
                      figsize=(8, 5), sharey=False)
fig.show()


# map
figM, axM = plt.subplots(1, 1, figsize=(8, 4))
siteNoLstCode = dictSite[code]
indS = [siteNoLst.index(siteNo) for siteNo in siteNoLstCode]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstCode)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
shortName = usgs.codePdf.loc[code]['shortName']
matMap = corrLSTM[indS, 1]
axplot.mapPoint(axM, lat, lon, matMap, s=24)
figM.show()

# ts map


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(12, 4))
    axplot.mapPoint(axM, lat, lon, rmseLSTM, s=24)
    axM.set_title('Testing corr LSTM ')
    figP = plt.figure(figsize=[16, 4])
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLstCode[iP]
    outName1 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QTFP_C', trainSet)
    dfL1 = basins.loadSeq(outName1, siteNo)
    dfO = waterQuality.readSiteTS(siteNo, [code], freq='W')
    t = dfO.index
    # ts
    tBar = np.datetime64('2010-01-01')
    sd = np.datetime64('1980-01-01')
    legLst = ['LSTM', 'Obs']
    axplot.plotTS(axP, t, [dfL1[code],  dfO[code]],
                  tBar=tBar, sd=sd, styLst='-*', cLst='rk', legLst=legLst)
    axP.set_title('site {} corr={:.3f}'.format(siteNo, matMap[iP]))
    axP.legend()


matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

importlib.reload(axplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)

saveFolder = r'C:\Users\geofk\work\Presentation\AGU2020'
# circled map
tempLst = ['01634000', '08332010']
# tempLst = ['11074000', '06902000', '01674500']
siteNoLstCode = dictSite[code]
indS = [siteNoLst.index(siteNo) for siteNo in siteNoLstCode]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstCode)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
shortName = usgs.codePdf.loc[code]['shortName']

figM, axM = plt.subplots(1, 1, figsize=(8, 4))
matMap = corrLSTM[indS, 1]
axplot.mapPoint(axM, lat, lon, matMap, s=24)
for siteNo in tempLst:
    xLoc = lon[siteNoLstCode.index(siteNo)]
    yLoc = lat[siteNoLstCode.index(siteNo)]
    circle = plt.Circle([xLoc, yLoc], 1,
                        color='red', fill=False)
    axM.add_patch(circle)
figM.show()
figM.savefig(os.path.join(saveFolder, 'corrMap_{}'.format(code)))


figM, axM = plt.subplots(1, 1, figsize=(8, 4))
matMap = rmseLSTM[indS, 1]
axplot.mapPoint(axM, lat, lon, matMap, s=24)
for siteNo in tempLst:
    xLoc = lon[siteNoLstCode.index(siteNo)]
    yLoc = lat[siteNoLstCode.index(siteNo)]
    circle = plt.Circle([xLoc, yLoc], 1,
                        color='r', fill=False)
    axM.add_patch(circle)
figM.show()
figM.savefig(os.path.join(saveFolder, 'rmseMap_{}'.format(code)))

for siteNo in tempLst:
    figP, axP = plt.subplots(1, 1, figsize=(8, 2.5))
    outName1 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QTFP_C', trainSet)
    dfL1 = basins.loadSeq(outName1, siteNo)
    dfO = waterQuality.readSiteTS(siteNo, [code], freq='W')
    t = dfO.index
    # ts
    tBar = np.datetime64('2010-01-01')
    sd = np.datetime64('1980-01-01')
    legLst = ['LSTM', 'Obs']
    axplot.plotTS(axP, t, [dfL1[code],  dfO[code]],
                  tBar=tBar, sd=sd, styLst='-*', cLst='rk', legLst=legLst)
    iP = siteNoLst.index(siteNo)
    axP.set_title('water temperature of site {} RMSE={:.2f} corr={:.2f}'.format(
        siteNo, rmseLSTM[iP, 1], corrLSTM[iP, 1]))
    axP.legend()
    figP.show()
    figP.savefig(os.path.join(saveFolder, 'ts_{}_{}'.format(code, siteNo)))

np.nanmean(rmseLSTM[:,1])