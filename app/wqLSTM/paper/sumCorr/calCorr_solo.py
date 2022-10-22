
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


ep = 500
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
# label = 'FPRT2QC'
DF = dbBasin.DataFrameBasin('G200')
bQ = np.isnan(DF.q[:, :, 0])
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)

# LSTM comb
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, 1000)
corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, 1000)
corrFile1 = os.path.join(outFolder, corrName1)
corrFile2 = os.path.join(outFolder, corrName2)
corrL1 = np.load(corrFile1)
corrL2 = np.load(corrFile2)

# solo models
nt, ns, nc = DF.c.shape
reTest = False
corrS1 = np.full([ns, nc], np.nan)
corrS2 = np.full([ns, nc], np.nan)
for k, code in enumerate(DF.varC):
    outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
    yP, ycP = basinFull.testModel(
        outName, DF=DF, testSet='all', ep=300, reTest=reTest, batchSize=50)
    yP[bQ] = np.nan
    pred1 = DF.extractSubset(yP, trainSet)
    pred2 = DF.extractSubset(yP, testSet)
    corrS1[:, k] = utils.stat.calCorr(pred1[:, :, 0], obs1[:, :, k])
    corrS2[:, k] = utils.stat.calCorr(pred2[:, :, 0], obs2[:, :, k])
