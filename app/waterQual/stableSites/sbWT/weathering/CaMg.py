from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt

codeLst = ['00915', '00925','00955']
ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)

code = '00915'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
outName = '{}-{}-{}-{}'.format(dataName, code, 'ntnS', trainSet)
# outName = '{}-{}-{}-{}'.format(dataName, code, 'plain', trainSet)
siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
corrMat = np.full([len(siteNoLst),  2], np.nan)
rmseMat = np.full([len(siteNoLst),  2], np.nan)
ic = wqData.varC.index(code)
for iT, subset in enumerate([trainSet, testSet]):
    yP, ycP = basins.testModel(
        outName, subset, wqData=wqData, ep=ep, reTest=reTest)
    ind = wqData.subset[subset]
    info = wqData.info.iloc[ind].reset_index()
    o = wqData.c[-1, ind, ic]
    p = yP[-1, :, 1]
    for iS, siteNo in enumerate(siteNoLst):
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iT] = corr
        rmseMat[iS, iT] = rmse
massMat = np.full([len(siteNoLst)], np.nan)
info = wqData.info
for iS, siteNo in enumerate(siteNoLst):
    indS = info[info['siteNo'] == siteNo].index.values
    massMat[iS] = np.nanmean(wqData.c[-1, indS, ic])
    # massMat[iS] = np.nanstd(wqData.c[-1, indS, ic]) / \
    #     np.nanmean(wqData.c[-1, indS, ic])

# plot map
shortName = usgs.codePdf.loc[code]['shortName']
figM, axM = plt.subplots(3, 1, figsize=(6, 8))
axplot.mapPoint(axM[0], lat, lon, massMat, s=16)
axM[0].set_title('Average concentration of {}'.format(shortName))
axplot.mapPoint(axM[1], lat, lon, rmseMat[:, 1], s=16)
axM[1].set_title('Testing RMSE of {}'.format(shortName))
axplot.mapPoint(axM[2], lat, lon, corrMat[:, 1], vRange=[0.5, 1], s=16)
axM[2].set_title('Testing correlation of {}'.format(shortName))
plt.tight_layout()
figM.show()
