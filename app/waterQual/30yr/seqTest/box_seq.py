from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictNB_y16n36.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# outName = 'sbWT-00945-ntnS-00945-Y1'
outName = 'sbWT-00945-plain-00945-Y1'
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)
code = '00945'
siteNoLst = dictSite[code]
ep = None
retest = True
basins.testModelSeq(outName, siteNoLst, wqData=wqData)
rmseMat = np.ndarray([len(siteNoLst), 2])
corrMat = np.ndarray([len(siteNoLst), 2])
for k, siteNo in enumerate(siteNoLst):
    dfP = basins.loadSeq(outName, siteNo)
    dfO = waterQuality.readSiteTS(
        siteNo, dfP.columns.tolist(), freq=wqData.freq)
    codeLst = dfP.columns.tolist()
    codeLst = ['00945']
    sd = np.datetime64('1980-01-01')
    ed = np.datetime64('2020-12-31')
    dfP = dfP[dfP.index >= sd]
    dfO = dfO[dfO.index >= sd]
    yr = pd.DatetimeIndex(dfP.index).year
    dfP1 = dfP[yr % 2 == 1]
    dfO1 = dfO[yr % 2 == 1]
    dfP2 = dfP[yr % 2 == 0]
    dfO2 = dfO[yr % 2 == 0]
    rmse1, corr1 = utils.stat.calErr(dfP1[code].values, dfO1[code].values)
    rmse2, corr2 = utils.stat.calErr(dfP2[code].values, dfO2[code].values)
    rmseMat[k, :] = [rmse1, rmse2]
    corrMat[k, :] = [corr1, corr2]


rmseMat2 = np.ndarray([len(siteNoLst), 2])
corrMat2 = np.ndarray([len(siteNoLst), 2])
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
master = basins.loadMaster(outName)
ic = wqData.varC.index(code)
for iT, subset in enumerate([trainSet, testSet]):
    yP, ycP = basins.testModel(
        outName, subset, wqData=wqData)
    ind = wqData.subset[subset]
    info = wqData.info.iloc[ind].reset_index()
    if dataName == 'sbWT':
        o = wqData.c[-1, ind, ic]
        p = yP[-1, :, 1]
    elif dataName == 'sbW':
        o = wqData.c[ind, ic]
        p = ycP[:, 0]
    for iS, siteNo in enumerate(siteNoLst):
        indS = info[info['siteNo'] == siteNo].index.values
        if len(indS) > 0:
            [a, b], indV = utils.rmNan([o[indS], p[indS]])
            corr = np.corrcoef(a, b)[0, 1]
            rmse = np.sqrt(np.nanmean((a-b)**2))
            corrMat2[iS,  iT] = corr
            rmseMat2[iS,  iT] = rmse

fig, ax = plt.subplots(1, 1)
ax.plot(corrMat[:, 1], corrMat2[:, 1], '*')
fig.show()

np.nanmedian(corrMat[:, 1])
np.nanmedian(corrMat2[:, 1])
