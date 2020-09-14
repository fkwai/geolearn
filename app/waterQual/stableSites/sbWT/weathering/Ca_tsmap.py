from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)

code = '00915'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
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

# plot ts


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(6, 4))
    axplot.mapPoint(axM, lat, lon, corrMat[:, 1], vRange=[0.5, 1], s=16)
    shortName = usgs.codePdf.loc[code]['shortName']
    axM.set_title('Testing correlation of {}'.format(shortName))
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfP = basins.loadSeq(outName, siteNo)[code]
    dfO = waterQuality.readSiteTS(siteNo, [code], freq=wqData.freq)[code]
    yr = pd.DatetimeIndex(dfP.index).year
    dfO1 = dfO[yr % 2 == 1]
    dfO2 = dfO[yr % 2 == 0]
    axplot.plotTS(axP, dfP.index, dfP.values, styLst='-', cLst='b')
    axplot.plotTS(axP, dfO1.index, dfO1.values, styLst='*', cLst='m')
    axplot.plotTS(axP, dfO2.index, dfO2.values, styLst='*', cLst='r')
    axP.legend(['pred','obs train','obs test'])
    dfC = pd.DataFrame(index=dfO2.dropna().index)
    dfC['obs'] = dfO2
    dfC['pred'] = dfP
    rmse, corr = utils.stat.calErr(dfC['pred'].values, dfC['obs'].values)
    axP.set_title('site {} corr = {:.3f}'.format(siteNo, corr))


figM, figP = figplot.clickMap(funcMap, funcPoint)
