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

code = '00955'

trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
labelLst = ['plain', 'ntn', 'ntnq']
siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()
corrMat = np.full([len(siteNoLst), len(labelLst)], np.nan)
rmseMat = np.full([len(siteNoLst),  len(labelLst)], np.nan)
for iLab, label in enumerate(labelLst):
    outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
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


def funcMap():
    figM, axM = plt.subplots(3, 1, figsize=(6, 8))
    axplot.mapPoint(axM[0], lat, lon, corrMat[:, 2], vRange=[0.5, 1], s=16)
    axplot.mapPoint(axM[1], lat, lon, corrMat[:, 1]/corrMat[:, 0], s=16)
    axplot.mapPoint(axM[2], lat, lon, corrMat[:, 2]/corrMat[:, 1], s=16)
    shortName = usgs.codePdf.loc[code]['shortName']
    # axM.set_title('Testing correlation of {}'.format(shortName))
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    cLst = 'cgb'
    dfO = waterQuality.readSiteTS(siteNo, [code], freq=wqData.freq)[code]
    yr = pd.DatetimeIndex(dfO.index).year
    dfO1 = dfO[yr % 2 == 1]
    dfO2 = dfO[yr % 2 == 0]
    dfC = pd.DataFrame(index=dfO2.dropna().index)
    dfC['obs'] = dfO2
    for k, label in enumerate(labelLst):
        outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
        dfP = basins.loadSeq(outName, siteNo)[code]
        dfC[label] = dfP
        axplot.plotTS(axP, dfP.index, dfP.values, styLst='-', cLst=cLst[k])
    axplot.plotTS(axP, dfO1.index, dfO1.values, styLst='*', cLst='m')
    axplot.plotTS(axP, dfO2.index, dfO2.values, styLst='*', cLst='r')
    axP.legend(labelLst+['obs train', 'obs test'])
    rmse1, corr1 = utils.stat.calErr(dfC['ntn'].values, dfC['obs'].values)
    rmse2, corr2 = utils.stat.calErr(dfC['ntnq'].values, dfC['obs'].values)
    axP.set_title(
        'site {} corr ntn = {:.3f} corr ntn+q = {:.3f}'.format(siteNo, corr1, corr2))


figM, figP = figplot.clickMap(funcMap, funcPoint)
