from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictSB_0412.json')) as f:
    dictSite = json.load(f)
ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)

code = '00915'

trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
labelLst = ['ntn', 'ntnq']
siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()
corrMat = np.full([len(siteNoLst), len(labelLst)], np.nan)
rmseMat = np.full([len(siteNoLst),  len(labelLst)], np.nan)
corrMatQ = np.full([len(siteNoLst)], np.nan)

dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
for iLab, label in enumerate(labelLst):
    outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
    ic = wqData.varC.index(code)
    subset = testSet
    yP, ycP = basins.testModel(
        outName, subset, wqData=wqData, ep=ep, reTest=reTest)
    ind = wqData.subset[subset]
    info = wqData.info.iloc[ind].reset_index()
    o = wqData.c[-1, ind, ic]
    if label == 'ntnq':
        p = yP[-1, :, 0]
    else:
        p = yP[-1, :, 1]
    for iS, siteNo in enumerate(siteNoLst):
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iLab] = corr
        rmseMat[iS, iLab] = rmse
        if label != 'ntnq':
            rmse, corr = utils.stat.calErr(
                yP[-1, indS, 0], wqData.q[-1, ind[indS], 0])
            corrMatQ[iS] = corr

figM, axM = plt.subplots(3, 1, figsize=(6, 8))
axplot.mapPoint(axM[0], lat, lon, corrMat[:, 1], vRange=[0.5, 1], s=16)
axplot.mapPoint(axM[1], lat, lon, corrMat[:, 1]/corrMat[:, 0], s=16)
axplot.mapPoint(axM[2], lat, lon, corrMatQ, s=16)
figM.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 8))
ax.plot(corrMat[:, 1]/corrMat[:, 0],corrMatQ,'*')
fig.show()
