
import scipy
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
codeLst = usgs.varC


# LSTM
DF = dbBasin.DataFrameBasin('G200')

ep = 1000
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
# label = 'FPRT2QC'
# trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
# testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']
trainLst = ['rmR20', 'rmL20', 'rmYr5']
testLst = ['pkR20', 'pkL20', 'pkYr5']

pdfP = pd.DataFrame(index=codeLst, columns=trainLst)
for trainSet, testSet in zip(trainLst, testLst):
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    outFolder = basinFull.nameFolder(outName)
    corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
    corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
    corrFile1 = os.path.join(outFolder, corrName1)
    corrFile2 = os.path.join(outFolder, corrName2)
    corrL1 = np.load(corrFile1)
    corrL2 = np.load(corrFile2)

    # WRTDS
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
    corrName1 = 'corr-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
    corrName2 = 'corr-{}-{}-{}.npy'.format('G200N', testSet, testSet)
    corrFile1 = os.path.join(dirWRTDS, corrName1)
    corrFile2 = os.path.join(dirWRTDS, corrName2)
    corrW1 = np.load(corrFile1)
    corrW2 = np.load(corrFile2)

    # count
    matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
            ).astype(int).astype(float)
    matB1 = DF.extractSubset(matB, trainSet)
    matB2 = DF.extractSubset(matB, testSet)
    count1 = np.nansum(matB1, axis=0)
    count2 = np.nansum(matB2, axis=0)
    matRm = (count1 < 80) | (count2 < 20)
    for corr in [corrL1, corrL2, corrW1, corrW2]:
        corr[matRm] = np.nan

    # load linear/seasonal
    dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirPar, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    # re-order

    codeStrLst = list()
    pLst = list()
    for k, code in enumerate(codeLst):
        [a, b], _ = utils.rmNan([corrL2[:, k], corrW2[:, k]])
        s, p = scipy.stats.wilcoxon(a, b)
        pdfP.at[code, trainSet] = p
pdfP.to_csv('temp.csv', sep=',', float_format='{:.2e}'.format())
