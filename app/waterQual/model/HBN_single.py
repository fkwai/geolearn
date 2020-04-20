import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('HBN')

doLst = list()
doLst.append('subset')

if 'subset' in doLst:
    # find ind have SiO4, P
    codeLst = ['00618', '00955']
    icLst = [wqData.varC.index(code) for code in codeLst]
    indAll = np.where(~np.isnan(wqData.c[:, icLst]).all(axis=1))[0]
    indAny = np.where(~np.isnan(wqData.c[:, icLst]).any(axis=1))[0]
    wqData.saveSubset('-'.join(sorted(codeLst)+['all']), indAll)
    wqData.saveSubset('-'.join(sorted(codeLst)+['any']), indAny)
    # seperate index by years
    for ind, lab in zip([indAll, indAny], ['all', 'any']):
        indYr1 = waterQuality.indYr(
            wqData.info.iloc[ind], yrLst=[1979, 2000])[0]
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y8090']), indYr1)
        indYr2 = waterQuality.indYr(
            wqData.info.iloc[ind], yrLst=[2000, 2020])[0]
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y0010']), indYr2)
    for code in codeLst:
        ic = wqData.varC.index(code)
        indC = np.where(~np.isnan(wqData.c[:, ic]))[0]
        indYr1 = waterQuality.indYr(
            wqData.info.iloc[indC], yrLst=[1979, 2000])[0]
        wqData.saveSubset(code+'-Y8090', indYr1)
        indYr2 = waterQuality.indYr(
            wqData.info.iloc[indC], yrLst=[2000, 2020])[0]
        wqData.saveSubset(code+'-Y0010', indYr2)
    # d=wqData.info.iloc[wqData.subset['00618-00955-any-Y10']]['date']
    # np.sort(pd.DatetimeIndex(d).year.unique())
    # ind=wqData.info.iloc[wqData.subset['00618-00955-any-Y10']].index.values
    # wqData.c[ind, wqData.varC.index('00618')]

# test
codeLst = ['00618', '00955']
varPred = ['00060']+codeLst
dataName = 'HBN'
outName = 'HBN-00618-00955-all-Y8090-opt2'
trainset = '00618-00955-all-Y8090'
testset = '00618-00955-all-Y0010'
# point test
yP1, ycP1 = basins.testModel(outName, trainset, wqData=wqData)
errMat1 = wqData.errBySiteC(ycP1, codeLst, subset=trainset)
yP2, ycP2 = basins.testModel(outName, testset, wqData=wqData)
errMat2 = wqData.errBySiteC(ycP2, codeLst, subset=testset)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
basins.testModelSeq(outName, siteNoLst, wqData=wqData)

# plot
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(2, 2, figsize=(8, 6))
    # shortName = codePdf.loc[code]['shortName']
    # title = 'RMSE of {} {}'.format(shortName, code)
    axplot.mapPoint(axM[0, 0], lat, lon, errMat1[:, 0, 0], s=12)
    axplot.mapPoint(axM[1, 0], lat, lon, errMat1[:, 1, 0], s=12)
    axplot.mapPoint(axM[0, 1], lat, lon, errMat2[:, 0, 0], s=12)
    axplot.mapPoint(axM[1, 1], lat, lon, errMat2[:, 1, 0], s=12)
    figP, axP = plt.subplots(3, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPred, dfObs = basins.loadSeq(outName, siteNo)
    t = dfPred['date'].values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')
    for k, var in enumerate(varPred):
        if var == '00060':
            styLst = '--'
            title = 'streamflow'
        else:
            styLst = '-*'
            shortName = codePdf.loc[var]['shortName']
            title = ' {} {}'.format(shortName, var)
        axplot.plotTS(axP[k], t, [dfPred[var], dfObs[var]], tBar=tBar,
                      legLst=['pred', 'obs'], styLst=styLst, cLst='br')
        axP[k].set_title(title)


figplot.clickMap(funcMap, funcPoint)
