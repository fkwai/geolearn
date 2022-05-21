
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
from hydroDL.post import adjustText
DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC

# LSTM
ep = 1000
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
dataNameLst = ['G200', 'G200N']
corrLst1 = list()
corrLst2 = list()
for dataName in dataNameLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    outFolder = basinFull.nameFolder(outName)
    corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
    corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
    corrFile1 = os.path.join(outFolder, corrName1)
    corrFile2 = os.path.join(outFolder, corrName2)
    corrL1 = np.load(corrFile1)
    corrL2 = np.load(corrFile2)
    corrLst1.append(corrL1)
    corrLst2.append(corrL2)


# WRTDS
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
corrName1 = 'corr-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
corrName2 = 'corr-{}-{}-{}.npy'.format('G200N', testSet, testSet)
corrFile1 = os.path.join(dirWRTDS, corrName1)
corrFile2 = os.path.join(dirWRTDS, corrName2)
corrW1 = np.load(corrFile1)
corrW2 = np.load(corrFile2)

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)
for corr in [corrW1, corrW2]+corrLst1+corrLst2:
    corr[matRm] = np.nan

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan


##
var1 = ['00010', '00095', '00300', '00915', '00925', '00930', '00935',
        '00940', '00945', '00955']

var2 = ['00405', '00600', '00605', '00618', '00660', '00665',
        '00681', '71846', '80154']
dataLst = [corrW2, corrLst2[0]]
styLst = ['b-*', 'r-*']
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
x = np.nanmean(matLR, axis=0)
for k, data in enumerate(dataLst):
    y = np.nanmean(data**2, axis=0)
    if k == 0:
        txtLst = list()
        for code in var1:
            ic = codeLst.index(code)
            txt = ax.text(x[ic], y[ic], usgs.codePdf.loc[code]['shortName'])
            txtLst.append(txt)
        for code in var2:
            ic = codeLst.index(code)
            txt = ax.text(x[ic], y[ic], usgs.codePdf.loc[code]['shortName'])
            txtLst.append(txt)
        adjustText.adjust_text(txtLst)
    ic1 = np.array([codeLst.index(code) for code in var1])
    ic2 = np.array([codeLst.index(code) for code in var2])

    ind1 = np.argsort(x[ic1])
    ind2 = np.argsort(x[ic2])
    ax.plot(x[ic1[ind1]], y[ic1[ind1]], styLst[k])
    ax.plot(x[ic2[ind2]], y[ic2[ind2]], styLst[k])
ax.plot([0, 1], [0, 1], 'k-')
fig.show()
figFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
fig.savefig(os.path.join(figFolder, 'twoDim_{}_{}'.format(label, trainSet)))
