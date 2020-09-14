from hydroDL import kPath, utils
from hydroDL.app import waterQuality, DGSA
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time

ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)

codeLst = usgs.codeLst.copy()
codeLst.remove('00410')
dfP = pd.DataFrame(index=gageII.varLst, columns=codeLst)
for code in codeLst:
    print(code)
    trainSet = '{}-Y1'.format(code)
    testSet = '{}-Y2'.format(code)
    outName = '{}-{}-{}-{}'.format(dataName, code, 'ntnp', trainSet)
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

    dfG = gageII.readData(varLst=gageII.varLst, siteNoLst=siteNoLst)
    dfG = gageII.updateCode(dfG)

    pMat = dfG.values
    dfS = DGSA.DGSA_light(
        pMat, corrMat[:, 1:2], ParametersNames=dfG.columns.tolist(), n_clsters=3)
    dfP[code] = dfS

importlib.reload(axplot)
dfP = dfP.sort_index(axis=1)
labX = list()
for code in dfP.columns.tolist():
    temp = usgs.codePdf.loc[code]['shortName']
    labX.append('{} {}'.format(temp, code))

labLst = [dfP.index.tolist(), labX]
fig, ax = plt.subplots()

ax = axplot.plotHeatMap(
    ax, dfP.values, fmt='{:.2f}', labLst=labLst, vRange=[0, 3])
fig.tight_layout()
fig.show()
