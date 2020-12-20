
import matplotlib
from astropy.timeseries import LombScargle
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

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
corrMat = np.full([nSite, len(codeLst), 2], np.nan)

# LSTM
label = 'QTFP_C'
trainSet = 'comb-B10'
testSet = 'comb-A10'
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
master = basins.loadMaster(outName)
subset = testSet
yP, ycP = basins.testModel(
    outName, subset, wqData=wqData, ep=ep, reTest=reTest)
ind = wqData.subset[subset]
info = wqData.info.iloc[ind].reset_index()
for iCode, code in enumerate(codeLst):
    ic = wqData.varC.index(code)
    if len(wqData.c.shape) == 3:
        p = yP[-1, :, master['varY'].index(code)]
        o = wqData.c[-1, ind, ic]
    elif len(wqData.c.shape) == 2:
        p = ycP[:, master['varYC'].index(code)]
        o = wqData.c[ind, ic]
    for siteNo in dictSite[code]:
        iS = siteNoLst.index(siteNo)
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iCode, 0] = corr
        # rmseMat[iS, iCode, iT*2] = rmse

# WRTDS
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'B10')
file2 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'A10N5'))
dfCorr2 = pd.read_csv(file2, dtype={'siteNo': str}).set_index('siteNo')
for iCode, code in enumerate(codeLst):
    indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
    corrMat[indS, iCode, 1] = dfCorr2.iloc[indS][code].values

# plot ts
code = ['00915']
iCode = codeLst.index(code)
indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
siteNoLstCode = dictSite[code]
# matMap = corrMat[indS, iCode, 0]-corrMat[indS, iCode, 1]
matMap = corrMat[indS, iCode, 0]
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
    dfW = pd.read_csv(os.path.join(dirWrtds, 'output', siteNo),
                      index_col=None).set_index('date')
    dfO = waterQuality.readSiteTS(siteNo, codeLst+['00060'], freq=wqData.freq)
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
