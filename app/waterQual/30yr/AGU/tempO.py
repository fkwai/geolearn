
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


matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

codePlot = ['00010', '00300']
figM, axM = plt.subplots(1, 2, figsize=(12, 4))
for k in range(2):
    code = codePlot[k]
    iCode = codeLst.index(code)
    siteNoLstCode = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoLstCode]
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstCode)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
    shortName = usgs.codePdf.loc[code]['shortName']
    matMap = corrMat[indS, iCode, 0]
    axplot.mapPoint(axM[k], lat, lon, matMap, vRange=[0, 1], s=16)
    # axM[k].set_title('Testing correlation of {} {}'.format(code, shortName))
figM.show()
