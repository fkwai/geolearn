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

dataName = 'basinAll'
wqData = waterQuality.DataModelWQ('basinAll')

outName = 'basinAll-Y8090-opt1'
trainset = 'Y8090'
testset = 'Y0010'
# point test
outFolder = os.path.join(kPath.dirWQ, 'model', outName)
yP1, ycP1 = basins.testModel(outName, trainset, wqData=wqData, ep=200)
errFile1 = os.path.join(outFolder, 'errMat1_ep200.npy')
# errMat1 = wqData.errBySiteC(ycP1, subset=trainset, varC=wqData.varC)
# np.save(errFile1, errMat1)
errMat1 = np.load(errFile1)

errFile2 = os.path.join(outFolder, 'errMat2_ep200.npy')
yP2, ycP2 = basins.testModel(outName, testset, wqData=wqData, ep=200)
# errMat2 = wqData.errBySiteC(ycP2, subset=testset, varC=wqData.varC)
# np.save(errFile2, errMat2)
errMat2 = np.load(errFile2)

# seq test
siteNoLst = wqData.info['siteNo'].unique().tolist()
# basins.testModelSeq(outName, siteNoLst, wqData=wqData, ep=200)

# figure out number of samples
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
df0 = pd.read_csv(os.path.join(dirInv, 'codeCount.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')
df1 = pd.read_csv(os.path.join(dirInv, 'codeCount_B2000.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')
df2 = pd.read_csv(os.path.join(dirInv, 'codeCount_A2000.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')
matN = df0.loc[siteNoLst].values
matN1 = df1.loc[siteNoLst].values
matN2 = df2.loc[siteNoLst].values

# plot box
codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    codeLst = codePdf[codePdf.group == group].index.tolist()
    indLst = [wqData.varC.index(code) for code in codeLst]
    labLst1 = [codePdf.loc[code]['shortName'] +
               '\n'+code for code in codeLst]
    labLst2 = ['train opt1', 'test opt1', 'train opt2', 'test opt2']
    dataBox = list()
    for ic in indLst:
        temp = list()
        for errMat in [errMat1, errMat2]:
            ind = np.where((matN1[:, ic] > 50) & (matN2[:, ic] > 50))[0]
            temp.append(errMat[ind, ic, 1])
        dataBox.append(temp)
    title = 'correlation of {} group'.format(group)
    fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
    fig.suptitle(title)
    fig.show()


# plot map
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
codePdf = usgs.codePdf

codeLst = ['00940', '00915']


def funcMap():
    nM = len(codeLst)
    figM, axM = plt.subplots(nM, 1, figsize=(8, 6))
    for k in range(0, nM):
        code = codeLst[k]
        ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = '{} {}'.format(shortName, code)
        axplot.mapPoint(axM[k], lat, lon, errMat2[:, ic, 1], s=12)
        axM[k].set_title(title)
    figP, axP = plt.subplots(nM+1, 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfPred, dfObs = basins.loadSeq(outName, siteNo, ep=200)
    dfPred = dfPred[dfPred.index >= np.datetime64('1980-01-01')]
    dfObs = dfObs[dfObs.index >= np.datetime64('1980-01-01')]
    t = dfPred.index.values.astype(np.datetime64)
    tBar = np.datetime64('2000-01-01')
    axplot.plotTS(axP[0], t, [dfPred['00060'], dfObs['00060']], tBar=tBar,
                  legLst=['pred', 'obs'], styLst='--', cLst='br')
    axP[0].set_title('streamflow')
    for k, var in enumerate(codeLst):
        styLst = '-*'
        shortName = codePdf.loc[var]['shortName']
        title = ' {} {}'.format(shortName, var)
        axplot.plotTS(axP[k+1], t, [dfPred[var], dfObs[var]], tBar=tBar,
                      legLst=['pred', 'obs'], styLst=styLst, cLst='br')
        axP[k+1].set_title(title)


figplot.clickMap(funcMap, funcPoint)
