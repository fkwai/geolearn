from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt

codeLst = ['00600', '00605', '00618', '71846']
ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)
corrLst = list()
rmseLst = list()
for code in codeLst:
    trainSet = '{}-Y1'.format(code)
    testSet = '{}-Y2'.format(code)
    if code == '00618':
        outName = '{}-{}-{}-{}'.format(dataName, code, 'ntnS', trainSet)
    else:
        outName = '{}-{}-{}-{}'.format(dataName, code, 'plain', trainSet)
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
    corrLst.append(corrMat)
    rmseLst.append(rmseMat)

# plot map
figM, axM = plt.subplots(2, 2, figsize=(8, 4))
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 2, 2)
    trainSet = '{}-Y1'.format(code)
    siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
    shortName = usgs.codePdf.loc[code]['shortName']
    axplot.mapPoint(axM[j,i], lat, lon, corrLst[k][:, 1], vRange=[0.0, 1], s=16)
    axM[j,i].set_title('Testing correlation of {}'.format(shortName))
    # axplot.mapPoint(axM[k], lat, lon, rmseLst[k][:, 1], s=16)
    # axM[k].set_title('Testing RMSE of {}'.format(shortName))
plt.tight_layout()
figM.show()
