from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# ts map of single dataset, label and code
ep = 500
dataName = 'nbW'
label = 'QT_C'
code = '00955'
trainSet = '{}-B16'.format('comb')
testSet = '{}-A16'.format('comb')
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictNB_y16n36.json')) as f:
    dictSite = json.load(f)
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = dictSite[code]
nSite = len(siteNoLst)
corrMat = np.full([nSite, 2], np.nan)
rmseMat = np.full([nSite, 2], np.nan)
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

# plot ts

dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def funcMap():
    figM, axM = plt.subplots(1, 2, figsize=(12, 4))
    axplot.mapPoint(axM[0], lat, lon, corrMat[:, 0], vRange=[0, 1], s=16)
    axplot.mapPoint(axM[1], lat, lon, corrMat[:, 1], vRange=[0, 1], s=16)
    shortName = usgs.codePdf.loc[code]['shortName']
    axM[0].set_title('Training correlation of {}'.format(shortName))
    axM[1].set_title('Testing correlation of {}'.format(shortName))
    figP, axP = plt.subplots(2, 1, figsize=(16, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfP = basins.loadSeq(outName, siteNo)[code]
    dfO = waterQuality.readSiteTS(siteNo, [code], freq=wqData.freq)[code]
    t = dfP.index
    yr = pd.DatetimeIndex(t).year
    dfO1 = dfO[yr <= 2016]
    dfO2 = dfO[yr > 2016]
    dfP1 = dfP[yr <= 2016]
    dfP2 = dfP[yr > 2016]
    axplot.plotTS(axP[0], dfP1.index, [dfP1.values, dfO1.values],
                  styLst='-*', cLst='br')
    axplot.plotTS(axP[1], dfP2.index, [dfP2.values, dfO2.values],
                  styLst='-*', cLst='br')
    # axP.legend(['pred', 'obs train', 'obs test'])
    rmse, corr = utils.stat.calErr(dfP1.values, dfO1.values)
    axP[0].set_title('site {} {:.2f} {:.2f}'.format(
        siteNo, corr, corrMat[iP, 0]))
    rmse, corr = utils.stat.calErr(dfP2.values, dfO2.values)
    axP[1].set_title('site {} {:.2f} {:.2f}'.format(
        siteNo, corr, corrMat[iP, 1]))


figM, figP = figplot.clickMap(funcMap, funcPoint)
