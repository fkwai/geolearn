
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

dataNameLst = ['G200N', 'G200']
labelLst = ['FPRT2QC', 'QFPRT2C', 'QFRT2C', 'QFPT2C', 'QT2C']
trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']

dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')

DF = dbBasin.DataFrameBasin('G200')
matObs = DF.c
for trainSet, testSet in zip(trainLst, testLst):
    fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
    yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']
    obs1 = DF.extractSubset(matObs, trainSet)
    obs2 = DF.extractSubset(matObs, testSet)
    pred1 = DF.extractSubset(yW, trainSet)
    pred2 = DF.extractSubset(yW, testSet)
    corr1 = utils.stat.calNash(pred1, obs1)
    corr2 = utils.stat.calNash(pred2, obs2)
    dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
    corrName1 = 'nash-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
    corrName2 = 'nash-{}-{}-{}.npy'.format('G200N', testSet, testSet)
    corrFile1 = os.path.join(dirRoot, corrName1)
    corrFile2 = os.path.join(dirRoot, corrName2)
    np.save(corrFile1, corr1)
    np.save(corrFile2, corr2)
