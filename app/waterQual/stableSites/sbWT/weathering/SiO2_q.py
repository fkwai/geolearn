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
labelLst = ['ntn', 'ntnq']
siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()
corrMat = np.full([len(siteNoLst), len(labelLst)], np.nan)
rmseMat = np.full([len(siteNoLst),  len(labelLst)], np.nan)
for iLab, label in enumerate(labelLst):
    outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
    master = basins.loadMaster(outName)
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
    p = yP[-1, :, master['varY'].index(code)]
    for iS, siteNo in enumerate(siteNoLst):
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iLab] = corr
        rmseMat[iS, iLab] = rmse


def funcMap():
    figM, axM = plt.subplots(1, 2, figsize=(10, 6))
    axplot.mapPoint(axM[0], lat, lon, corrMat[:, 1], vRange=[0.5, 1], s=16)
    axM[0].set_title('correlation with Q')
    diff = corrMat[:, 1]**2/corrMat[:, 0]**2
    axplot.mapPoint(axM[1], lat, lon, diff, s=16)
    axM[1].set_title('Rsq Q target / Rsq Q input')
    shortName = usgs.codePdf.loc[code]['shortName']
    # axM.set_title('Testing correlation of {}'.format(shortName))
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    cLst = 'cb'
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
        'site {} corr Q target = {:.3f} corr Q input = {:.3f}'.format(siteNo, corr1, corr2))

plt.tight_layout()
figM, figP = figplot.clickMap(funcMap, funcPoint)
