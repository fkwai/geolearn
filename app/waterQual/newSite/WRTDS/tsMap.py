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
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictNB_y16n36.json')) as f:
    dictSite = json.load(f)


ep = 500
dataName = 'nbW'
label = 'QT_C'
code = '00955'
trainSet = '{}-B16'.format('comb')
testSet = '{}-A16'.format('comb')
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = dictSite[code]
nSite = len(siteNoLst)
corrMat = np.full([nSite, 4], np.nan)
rmseMat = np.full([nSite, 4], np.nan)
siteNoLst = dictSite[code]

# LSTM
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
master = basins.loadMaster(outName)
for iT, subset in enumerate([trainSet, testSet]):
    yP, ycP = basins.testModel(
        outName, subset, wqData=wqData, ep=ep)
    ind = wqData.subset[subset]
    info = wqData.info.iloc[ind].reset_index()
    ic = wqData.varC.index(code)
    if len(wqData.c.shape) == 3:
        p = yP[-1, :, master['varY'].index(code)]
        o = wqData.c[-1, ind, ic]
    elif len(wqData.c.shape) == 2:
        p = ycP[:, master['varYC'].index(code)]
        o = wqData.c[ind, ic]
    for iS, siteNo in enumerate(siteNoLst):
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iT] = corr
        rmseMat[iS, iT] = rmse

# WRTDS
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS')
file1 = os.path.join(dirWrtds, '{}-{}-corr'.format('B16', 'B16'))
dfCorr1 = pd.read_csv(file1, dtype={'siteNo': str}).set_index('siteNo')
corrMat[:, 2] = dfCorr1.loc[siteNoLst][code].values
file2 = os.path.join(dirWrtds, '{}-{}-corr'.format('B16', 'A16'))
dfCorr2 = pd.read_csv(file2, dtype={'siteNo': str}).set_index('siteNo')
corrMat[:, 3] = dfCorr2.loc[siteNoLst][code].values

# plot corr vs
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
x = corrMat[:, 1]
y = corrMat[:, 3]
vmin = np.nanmin([x, y])
vmax = np.nanmax([x, y])
ax.set_xlim([vmin, vmax])
ax.set_ylim([vmin, vmax])
ax.plot([vmin, vmax], [vmin, vmax], 'k-')
ax.plot(x, y, '*')
ax.set_aspect('equal', 'box')
ax.set_xlabel('LSTM testing corr')
ax.set_ylabel('WRTDS testing corr')
fig.show()


# plot ts
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def funcMap():
    figM, axM = plt.subplots(1, 2, figsize=(12, 3))
    axplot.mapPoint(axM[0], lat, lon, corrMat[:, 1], vRange=[0, 1], s=16)
    axplot.mapPoint(axM[1], lat, lon, corrMat[:, 3], vRange=[0, 1], s=16)
    shortName = usgs.codePdf.loc[code]['shortName']
    axM[0].set_title('LSTM testing corr of {}'.format(shortName))
    axM[1].set_title('WRTDS testing corr of {}'.format(shortName))
    figP, axP = plt.subplots(2, 1, figsize=(20, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfO = waterQuality.readSiteTS(siteNo, [code], freq=wqData.freq)[code]
    t = dfO.index
    yr = t.year.values
    ind1 = (yr <= 2016) & (yr >= 1980)
    ind2 = yr > 2016
    o1 = dfO[ind1].values
    o2 = dfO[ind2].values
    t1 = t[ind1]
    t2 = t[ind2]
    # LSTM
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    dfP = basins.loadSeq(outName, siteNo)[code]
    # WRTDS
    fileWrtds = os.path.join(dirWrtds, 'B16', siteNo)
    dfW = pd.read_csv(fileWrtds, index_col=None).set_index('date')[code]
    dfW.index = pd.to_datetime(dfW.index)
    v1 = [dfP[ind1].values, dfW[ind1].values, o1]
    v2 = [dfP[ind2].values, dfW[ind2].values, o2]
    axplot.plotTS(axP[0], t1, v1, styLst='--*', cLst='bgr')
    axplot.plotTS(axP[1], t2, v2, styLst='--*', cLst='bgr')
    # print corr
    rmseWRTDS1, corrWRTDS1 = utils.stat.calErr(dfW[ind1].values, o1)
    rmseLSTM1, corrLSTM1 = utils.stat.calErr(dfP[ind1].values, o1)
    axP[0].set_title('site {} WRTDS {:.2f} LSTM {:.2f}'.format(
        siteNo, corrWRTDS1, corrLSTM1))
    rmseWRTDS2, corrWRTDS2 = utils.stat.calErr(dfW[ind2].values, o2)
    rmseLSTM2, corrLSTM2 = utils.stat.calErr(dfP[ind2].values, o2)
    axP[1].set_title('site {} WRTDS {:.2f} LSTM {:.2f}'.format(
        siteNo, corrWRTDS2, corrLSTM2))


figM, figP = figplot.clickMap(funcMap, funcPoint)
