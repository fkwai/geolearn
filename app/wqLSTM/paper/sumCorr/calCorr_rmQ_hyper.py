
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

# dataNameLst = ['G200N', 'G200']
dataName = 'G200'

trainSet = 'rmYr5'
testSet = 'pkYr5'

label = 'QFPRT2C'

# calculate and save corr for all cases
DF = dbBasin.DataFrameBasin(dataName)
bQ = np.isnan(DF.q[:, :, 0])
matObs = DF.c

codeLst = usgs.varC
ep = 500
dictLst = list()
# rhoLst = [180, 365, 750, 1000]
# for rho in rhoLst:
# outName = '{}-{}-{}-rho{}'.format(dataName, label, trainSet, rho)

hsLst = [16, 64, 128, 512]
for hs in hsLst:
    outName = '{}-{}-{}-hs{}'.format(dataName, label, trainSet, hs)
    obs1 = DF.extractSubset(matObs, trainSet)
    obs2 = DF.extractSubset(matObs, testSet)
    corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
    corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
    print(outName)
    outFolder = basinFull.nameFolder(outName)
    corrFile1 = os.path.join(outFolder, corrName1)
    corrFile2 = os.path.join(outFolder, corrName2)
    yP, ycP = basinFull.testModel(
        outName, DF=DF, testSet='all', ep=ep)
    varY = basinFull.loadMaster(outName)['varY']
    yOut = np.ndarray(yP.shape)
    for k, var in enumerate(varY):
        temp = yP[:, :, k]
        temp[bQ] = np.nan
        yOut[:, :, k] = temp
    if varY[0] == '00060':
        yOut = yOut[:, :, 1:]
    pred1 = DF.extractSubset(yOut, trainSet)
    pred2 = DF.extractSubset(yOut, testSet)
    corr1 = utils.stat.calCorr(pred1, obs1)
    corr2 = utils.stat.calCorr(pred2, obs2)
    np.save(corrFile1, corr1)
    np.save(corrFile2, corr2)
