
import matplotlib.gridspec as gridspec
from hydroDL.post import axplot, figplot, mapplot
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS

import warnings

DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.newC

# trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
# testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']

dataName = 'G200'
trainSet = 'rmRT20'
testSet = 'pkRT20'
label = 'QFPRT2C'

outName = '{}-{}-{}'.format(dataName, label, trainSet)
yP, ycP = basinFull.testModel(
    outName, DF=DF, testSet=testSet, ep=500)
yL = yP


# WRTDS
# yW = WRTDS.testWRTDS(dataName, trainSet, testSet, codeLst)
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName+'N', trainSet, testSet)
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# train / test
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)

code = '00618'
indC = DF.varC.index(code)
c = DF.c[:, :, indC]
count = np.sum(~np.isnan(c), axis=0)
lat, lon = DF.getGeo()

# map of count
indS = np.where(count >= 200)[0]
siteNoLst = [DF.siteNoLst[ind] for ind in indS]
figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(
    figM, gsM[0, 0], lat[indS], lon[indS], count[indS], s=20, cb=True)
figM.show()

# box plot
dataPlot = list()
codePlot = ['00618', '00600', '00605', '71846']
labLst = [usgs.codePdf.loc[code]['shortName'] +
          '\n'+code for code in codePlot]
for code in codePlot:
    indC = codeLst.index(code)
    statL = utils.stat.calStat(yL[:, indS, indC], d2.Y[:, indS, indC])
    statW = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
    dataPlot.append([statL['Corr'], statW['Corr']])
fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(12, 4),
                            label2=['LSTM', 'WRTDS'],
                            label1=labLst)
fig.show()

# tsMap
code = '00618'
siteNoLst = [DF.siteNoLst[ind] for ind in indS]
indC = DF.varC.index(code)
statL = utils.stat.calStat(yL[:, indS, indC], d2.Y[:, indS, indC])
matR = statL['Corr']


def funcM():
    figM = plt.figure()
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(
        figM, gsM[0, 0], lat[indS], lon[indS], matR, s=20, cb=True)
    figP = plt.figure(figsize=[12, 4])
    gsP = gridspec.GridSpec(1, 1)
    axP = figP.add_subplot(gsP[0, 0])
    return figM, axM, figP, axP, lon[indS], lat[indS]


def funcP(iP, axP):
    t = DF.t
    dataPlot = [yL[:, indS[iP], indC],
                d2.Y[:, indS[iP], indC]]
    axplot.plotTS(axP, t, dataPlot, legLst=['LSTM', 'obs test'],
                  styLst='-*', cLst='kr')
    titleStr = 'site {}; test corr {:.2f}; # obs {}'.format(
        siteNoLst[iP], matR[iP], count[indS[iP]])
    axP.set_title(titleStr)


figM, figP = figplot.clickMap(funcM, funcP)
