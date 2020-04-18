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
    # print number of samples
    for code in codeLst:
        ic = wqData.varC.index(code)
        indC = np.where(~np.isnan(wqData.c[:, ic]))[0]
    # seperate index by years
    for ind, lab in zip([indAll, indAny], ['all', 'any']):
        indYr1 = waterQuality.indYr(wqData.info.iloc[ind], yrLst=[1979, 2000])[0]
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y8090']), indYr1)
        indYr2 = waterQuality.indYr(wqData.info.iloc[ind], yrLst=[2000, 2020])[0]
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y0010']), indYr2)
    for code in codeLst:
        ic = wqData.varC.index(code)
        indC = np.where(~np.isnan(wqData.c[:, ic]))[0]
        indYr1 = waterQuality.indYr(wqData.info.iloc[indC], yrLst=[1979, 2000])[0]
        wqData.saveSubset(code+'-Y8090', indYr1)
        indYr2 = waterQuality.indYr(wqData.info.iloc[indC], yrLst=[2000, 2020])[0]
        wqData.saveSubset(code+'-Y0010', indYr2)
    # d=wqData.info.iloc[wqData.subset['00618-00955-any-Y10']]['date']
    # np.sort(pd.DatetimeIndex(d).year.unique())
    # ind=wqData.info.iloc[wqData.subset['00618-00955-any-Y10']].index.values
    # wqData.c[ind, wqData.varC.index('00618')]

# train local



# test
code = '00955'
out = 'HBN-00955-rmY10'
trainSet = '00955-rmY10'
testSet = '00955-Y10'
p1, o1 = basins.testModel(out, trainSet, wqData=wqData)
p2, o2 = basins.testModel(out, testSet, wqData=wqData)
errMat1 = wqData.errBySite(p1, subset=trainSet, varC=[code])
errMat2 = wqData.errBySite(p2, subset=testSet, varC=[code])

# plot
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(2, 1, figsize=(8, 6))
    shortName = codePdf.loc[code]['shortName']
    title = 'correlation of {} {}'.format(shortName, code)
    axplot.mapPoint(axM[0], lat, lon, errMat1[:, 0, 1], s=12)
    axplot.mapPoint(axM[1], lat, lon, errMat2[:, 0, 1], s=12)
    figP, axP = plt.subplots(1, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    info1 = wqData.subsetInfo(trainSet)
    info2 = wqData.subsetInfo(testSet)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    t1 = info1['date'][ind1].values.astype(np.datetime64)
    t2 = info2['date'][ind2].values.astype(np.datetime64)
    t = np.concatenate([t1, t2])
    x = np.concatenate([p1[ind1], p2[ind2]])
    y = np.concatenate([o1[ind1], o2[ind2]])
    # tBar = t1[-1]+(t2[0]-t1[-1])/2
    tBar= np.datetime64('2010-01-01')
    axplot.plotTS(axP, t, [x, y], styLst='-*', tBar=tBar,
                  legLst=['pred', 'obs'])


figplot.clickMap(funcMap, funcPoint)
