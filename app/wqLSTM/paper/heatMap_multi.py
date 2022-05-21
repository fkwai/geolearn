
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

DF = dbBasin.DataFrameBasin('G200')
ep = 500
codeLst = usgs.varC

# count matrix
matB = (~np.isnan(DF.c)).astype(int).astype(float)
for trainSet, testSet in zip(trainLst, testLst):
    # trainSet = 'rmR20'
    # testSet = 'pkR20'
    matB1 = DF.extractSubset(matB, trainSet)
    matB2 = DF.extractSubset(matB, testSet)
    count1 = np.nansum(matB1, axis=0)
    count2 = np.nansum(matB2, axis=0)
    matRm = (count1 < 80) | (count2 < 20)
    corrLst1 = list()
    corrLst2 = list()
    for label in labelLst:
        for dataName in dataNameLst:
            outName = '{}-{}-{}'.format(dataName, label, trainSet)
            outFolder = basinFull.nameFolder(outName)
            corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
            corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
            corrFile1 = os.path.join(outFolder, corrName1)
            corrFile2 = os.path.join(outFolder, corrName2)
            corr1 = np.load(corrFile1)
            corr1[matRm] = np.nan
            corrLst1.append(corr1)
            corr2 = np.load(corrFile2)
            corr2[matRm] = np.nan
            corrLst2.append(corr2)

    # label name
    caseLst = list()
    for label in labelLst:
        labelStr = label.split('T')[0]
        for dataName in dataNameLst:
            if dataName[-1] == 'N':
                caseLst.append('{}-LN'.format(labelStr))
            else:
                caseLst.append(labelStr)

    # WRTDS
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
    corrName1 = 'corr-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
    corrName2 = 'corr-{}-{}-{}.npy'.format('G200N', testSet, testSet)
    corrFile1 = os.path.join(dirWRTDS, corrName1)
    corrFile2 = os.path.join(dirWRTDS, corrName2)
    corrW1 = np.load(corrFile1)
    corrW1[matRm] = np.nan
    corrLst1.append(corrW1)
    corrW2 = np.load(corrFile2)
    corrW2[matRm] = np.nan
    corrLst2.append(corrW2)
    caseLst.append('WRTDS')

    # plot
    figFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
    codeStrLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]

    matPlot = np.full([len(corrLst2), len(codeLst)], np.nan)
    for k, corr in enumerate(corrLst2):
        matPlot[k, :] = np.nanmean(corr, axis=0)
    fig, ax = plt.subplots(1, 1)
    axplot.plotHeatMap(ax, matPlot*100, labLst=[caseLst, codeStrLst])
    title = 'Median Testing Correlation of Models'
    ax.set_title(title)
    plt.tight_layout()
    fig.show()
    plt.savefig(os.path.join(
        figFolder, 'heatmap_AllModel_{}'.format(trainSet)))
