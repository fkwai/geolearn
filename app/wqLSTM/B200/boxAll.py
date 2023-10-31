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
ep = 500

# load global model
epG = 400  # models are killed before 500
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


# calculate stats
matObs = DFA.c
obs1 = DFA.extractSubset(matObs, trainSet)
obs2 = DFA.extractSubset(matObs, testSet)
corrG1 = list()
corrG2 = list()
corrW1 = list()
corrW2 = list()
importlib.reload(utils.stat)
# strFunc='calLogMAE'
strFunc='calCorr'
errFunc=getattr(utils.stat,strFunc)
for iC, code in enumerate(usgs.varC):
    corrGT1 = list()
    corrGT2 = list()
    for iL, label in enumerate(labelLst):
        corrGT1.append(errFunc(yGLst1[iL][:, :, iC], obs1[:, :, iC]))
        corrGT2.append(errFunc(yGLst2[iL][:, :, iC], obs2[:, :, iC]))        
    corrW1.append(errFunc(yW1[:, :, iC], obs1[:, :, iC]))
    corrW2.append(errFunc(yW2[:, :, iC], obs2[:, :, iC]))
    corrG1.append(corrGT1)
    corrG2.append(corrGT2)

# pred=yGLst1[iL][:, :, iC]
# obs=obs1[:, :, iC]
# np.exp(np.nanmean(np.abs(np.log(obs/pred))))

# utils.stat.calNash(pred,obs)

# errFunc(yGLst1[iL][:, :, iC], obs1[:, :, iC])

# box plot
mean = np.array([np.nanmean(corr) for corr in corrW2])
indPlot = np.argsort(mean)
codeStrLst = list()
dataPlot = list()
iL = 0
# for k in indPlot:
for k in range(len(usgs.varC)):
    code = usgs.varC[k]
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    dataPlot.append([corrW1[k], corrG1[k][iL]])  # training
    # dataPlot.append([corrW2[k], corrG2[k][iL]])  # testing
fig, axes = figplot.boxPlot(
    dataPlot,
    widths=0.5,
    figsize=(12, 4),    
    label1=codeStrLst,
    label2=['WRTDS', 'LSTM-global'],
)
fig.suptitle('{}, {}'.format(strFunc,labelLst[iL]))
fig.show()
