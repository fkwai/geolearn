
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
dataNameLst = ['G200']
labelLst = ['FPRT2QC', 'QFPRT2C', 'QFRT2C', 'QFPT2C', 'QT2C']
# trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
# testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']


# quick scan
dirModel = r'C:\Users\geofk\work\waterQuality\modelFull'
for dataName in dataNameLst:
    for label in labelLst:
        for trainSet in trainLst:
            outName = '{}-{}-{}'.format(dataName, label, trainSet)
            fileName = os.path.join(dirModel, outName, 'modelState_ep1000')
            if not os.path.isfile(fileName):
                print(outName)

# calculate and save corr for all cases
DF = dbBasin.DataFrameBasin('G200')
matObs = DF.c
bQ = np.isnan(DF.q[:, :, 0])
codeLst = usgs.varC
ep = 1000
dictLst = list()
for trainSet, testSet in zip(trainLst, testLst):
    obs1 = DF.extractSubset(matObs, trainSet)
    obs2 = DF.extractSubset(matObs, testSet)
    corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
    corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
    for dataName in dataNameLst:
        for label in labelLst:
            print(outName)
            outName = '{}-{}-{}'.format(dataName, label, trainSet)
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
