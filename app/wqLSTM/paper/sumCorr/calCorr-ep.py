
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

dataName = 'G200'
label = 'QFPRT2C'
trainSet = 'rmYr5'
testSet = 'pkYr5'

# calculate and save corr for all cases
DF = dbBasin.DataFrameBasin('G200')
matObs = DF.c
codeLst = usgs.newC
epLst = [500,  1500, 2000]
dictLst = list()
for ep in epLst:
    obs1 = DF.extractSubset(matObs, trainSet)
    obs2 = DF.extractSubset(matObs, testSet)
    corrName1 = 'corr-{}-Ep{}.npy'.format(trainSet, ep)
    corrName2 = 'corr-{}-Ep{}.npy'.format(testSet, ep)
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    outFolder = basinFull.nameFolder(outName)
    corrFile1 = os.path.join(outFolder, corrName1)
    corrFile2 = os.path.join(outFolder, corrName2)
    print(outName)
    yP, ycP = basinFull.testModel(
        outName, DF=DF, testSet='all', ep=ep)
    yOut = yP
    if label[0] is not 'Q':
        yOut = yOut[:, :, 1:]
    pred1 = DF.extractSubset(yOut, trainSet)
    pred2 = DF.extractSubset(yOut, testSet)
    corr1 = utils.stat.calCorr(pred1, obs1)
    corr2 = utils.stat.calCorr(pred2, obs2)
    np.save(corrFile1, corr1)
    np.save(corrFile2, corr2)
