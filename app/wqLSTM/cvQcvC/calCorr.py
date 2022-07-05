
import matplotlib
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import sklearn.tree
import matplotlib.gridspec as gridspec
from hydroDL.master import basinFull


# investigate CV(C) / CV(Q) as an indicator of model performance

codeLst = usgs.varC

DF = dbBasin.DataFrameBasin('G200')
# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)
matRm1 = count < 200

trainSet = 'rmYr5'
testSet = 'pkYr5'
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm2 = (count1 < 80) | (count2 < 20)

# CVc/CVq
matCV = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
q = DF.q[:, :, 1]
cvQ = np.nanstd(q, axis=0)/np.nanmean(q, axis=0)
for k, code in enumerate(codeLst):
    c = DF.c[:, :, k]
    cvC = np.nanstd(c, axis=0)/np.nanmean(c, axis=0)
    matCV[:, k] = cvC/cvQ


# LSTM
ep = 500
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
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

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(
        filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values

matLR1 = matLR.copy()
matCV1 = matCV.copy()
matLR1[matRm1] = np.nan
matCV1[matRm1] = np.nan
matLR2 = matLR.copy()
matCV2 = matCV.copy()
matLR2[matRm2] = np.nan
matCV2[matRm2] = np.nan
d = np.nanmean(corrL2**2 - corrW2**2, axis=0)
s1 = np.nanmean(matLR1, axis=0)
s2 = np.nanmean(matLR2, axis=0)
c1 = np.nanmean(matCV1, axis=0)
c2 = np.nanmean(matCV2, axis=0)

# calculate a coefficient
codeLst = DF.varC
codeCal = codeLst.copy()
codeCal.remove('00010')
codeCal.remove('00300')
codeCal.remove('00400')
codeCal.remove('00405')
codeCal.remove('80154')

ind = [codeLst.index(code) for code in codeCal]
np.corrcoef(s1[ind], c1[ind])
np.corrcoef(s2[ind], d[ind])
np.corrcoef(c2[ind], d[ind])
