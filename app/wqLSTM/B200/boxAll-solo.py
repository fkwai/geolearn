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
import matplotlib


labelLst = ['FT2QC', 'QFT2C', 'QT2C']
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
epG = 400  # models are killed before 500
epL = 100

# load global model
DFA = dbBasin.DataFrameBasin('rmTK-B200')
yGLst1 = list()
yGLst2 = list()
for label in labelLst:
    outName = '{}-{}-{}'.format('rmTK-B200', label, trainSet)
    dictMaster = basinFull.loadMaster(outName)
    yP1, ycP1 = basinFull.testModel(outName, DF=DFA, testSet=trainSet, ep=epG)
    yP2, ycP2 = basinFull.testModel(outName, DF=DFA, testSet=testSet, ep=epG)
    if dictMaster['varY'][0] == 'streamflow':
        yP1 = yP1[:, :, 1:]
        yP2 = yP2[:, :, 1:]
    yGLst1.append(yP1)
    yGLst2.append(yP2)

# load WRTDS
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('rmTK-B200', trainSet, 'all.npz')
yW = np.load(os.path.join(dirWRTDS, fileName))['yW']
yW1 = DFA.extractSubset(yW, trainSet)
yW2 = DFA.extractSubset(yW, testSet)


# local model
corrG1 = list()
corrG2 = list()
corrL1 = list()
corrL2 = list()
corrW1 = list()
corrW2 = list()
strFunc = 'calNash'
errFunc = getattr(utils.stat, strFunc)
for iC, code in enumerate(usgs.varC):
    corrLT1 = list()
    corrLT2 = list()
    corrGT1 = list()
    corrGT2 = list()
    dataName = '{}-{}'.format(code, 'B200')
    DF = dbBasin.DataFrameBasin(dataName)
    matObs = DF.extractT([code])
    obs1 = DF.extractSubset(matObs, trainSet)
    obs2 = DF.extractSubset(matObs, testSet)
    # WRTDS
    _, indS1, indS2 = utils.intersect(DF.siteNoLst, DFA.siteNoLst, returnIndex=True)
    _, indT1, indT2 = np.intersect1d(DF.t, DFA.t, return_indices=True)

    for iL, label in enumerate(labelLst):
        # local model
        outName = '{}-{}-{}'.format(dataName, label, trainSet)
        dictMaster = basinFull.loadMaster(outName)
        yP1, ycP1 = basinFull.testModel(outName, DF=DF, testSet=trainSet, ep=epL)
        yP2, ycP2 = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=epL)
        if len(dictMaster['varY']) > 1:
            yP1 = yP1[:, :, 1:]
            yP2 = yP2[:, :, 1:]
        corrLT1.append(errFunc(yP1[:, indS1, 0][indT1, :], obs1[:, indS1, 0][indT1, :]))
        corrLT2.append(errFunc(yP2[:, indS1, 0][indT1, :], obs2[:, indS1, 0][indT1, :]))
        corrGT1.append(
            errFunc(yGLst1[iL][:, indS2, iC][indT2, :], obs1[:, indS1, 0][indT1, :])
        )
        corrGT2.append(
            errFunc(yGLst2[iL][:, indS2, iC][indT2, :], obs2[:, indS1, 0][indT1, :])
        )
    corrW1.append(errFunc(yW1[:, indS2, iC][indT2, :], obs1[:, indS1, 0][indT1, :]))
    corrW2.append(errFunc(yW2[:, indS2, iC][indT2, :], obs2[:, indS1, 0][indT1, :]))
    corrL1.append(corrLT1)
    corrL2.append(corrLT2)
    corrG1.append(corrGT1)
    corrG2.append(corrGT2)


# box plot
mean = np.array([np.nanmean(corr) for corr in corrW2])
indPlot = np.argsort(mean)
codeStrLst = list()
dataPlot = list()
iL = 0
for k in indPlot:
    code = usgs.varC[k]
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    dataPlot.append([corrW1[k], corrL1[k][iL], corrG1[k][iL]])  # training
    # dataPlot.append([corrW2[k], corrL2[k][iL], corrG2[k][iL]])  # testing


fig, axes = figplot.boxPlot(
    dataPlot,
    widths=0.5,
    figsize=(12, 4),
    label1=codeStrLst,
    label2=['WRTDS', 'LSTM-solo', 'LSTM-global'],
)
fig.suptitle('{} {}'.format(strFunc, labelLst[iL]))
fig.show()
