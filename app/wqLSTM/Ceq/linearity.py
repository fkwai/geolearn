
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.app.waterQuality import WRTDS
import statsmodels.api as sm
import scipy
from hydroDL.app.waterQuality import cqType
import importlib
import time
import statsmodels.api as sm
from hydroDL.data import dbBasin, gageII, usgs
from hydroDL.master import basinFull
from astropy.timeseries import LombScargle

# load models
dataName = 'G200N'
DFN = dbBasin.DataFrameBasin(dataName)
codeLst = usgs.newC
trainSet = 'rmR20'
testSet = 'pkR20'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
yP, ycP = basinFull.testModel(
    outName, DF=DFN, testSet=testSet, ep=500)
yL = np.ndarray(yP.shape)
for k, code in enumerate(codeLst):
    m = DFN.g[:, DFN.varG.index(code+'-M')]
    s = DFN.g[:, DFN.varG.index(code+'-S')]
    yL[:, :, k] = yP[:, :, k]*s+m
siteNoLst = DFN.siteNoLst
ns = len(siteNoLst)
nc = len(codeLst)

# load WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


# correlation matrix
d1 = dbBasin.DataModelBasin(DFN, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DFN, subset=testSet, varY=codeLst)
siteNoLst = DFN.siteNoLst
matW = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
matL = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
for indS, siteNo in enumerate(siteNoLst):
    print(indS)
    for indC, code in enumerate(codeLst):
        n1 = np.sum(~np.isnan(d1.Y[:, indS, indC]), axis=0)
        n2 = np.sum(~np.isnan(d2.Y[:, indS, indC]), axis=0)
        if n1 >= 160 and n2 >= 40:
            statW = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
            matW[indS, indC, :] = list(statW.values())
            statL = utils.stat.calStat(yL[:, indS, indC], d2.Y[:, indS, indC])
            matL[indS, indC, :] = list(statL.values())

# linearity
sn = 1e-5
corrMat1 = np.full([len(DFN.siteNoLst), len(codeLst)], np.nan)
for indS, siteNo in enumerate(DFN.siteNoLst):
    for code in codeLst:
        indS = DFN.siteNoLst.index(siteNo)
        indC = DFN.varC.index(code)
        Q = DFN.q[:, indS, 1]
        C = DFN.c[:, indS, indC]
        q = np.log(Q+sn)
        c = np.log(C+sn)
        [x, y], _ = utils.rmNan([q, c])
        corrMat1[indS, indC] = np.corrcoef(x, y)[0, 1]

# seasonality
corrMat2 = np.full([len(siteNoLst), len(codeLst)], np.nan)
for indS, siteNo in enumerate(DFN.siteNoLst):
    for code in codeLst:
        indS = DFN.siteNoLst.index(siteNo)
        indC = DFN.varC.index(code)
        t = np.arange(len(DFN.t))
        y = DFN.c[:, indS, indC]
        tt, yy = utils.rmNan([t, y], returnInd=False)
        p = LombScargle(tt, yy).power(1/365)
        corrMat2[indS, indC] = p


# linearity vs correlation
corrMat = corrMat1
rMat = corrMat1**2
code = '00915'
indC = codeLst.index(code)
fig, ax = plt.subplots(1, 1)
axplot.scatter121(ax, matL[:, indC, 3], matW[:, indC, 3], corrMat[:, indC])
fig.show()

thR = 0.5
codePlot = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
dataBox = list()
for code in codePlot:
    indC = DFN.varC.index(code)
    ind1 = np.where(rMat[:, indC] <= thR)[0]
    ind2 = np.where(rMat[:, indC] > thR)[0]
    temp = [matL[ind1, indC, 3], matW[ind1, indC, 3],
            matL[ind2, indC, 3], matW[ind2, indC, 3]]
    dataBox.append(temp)

labLst1 = ['{}\n{}'.format(usgs.codePdf.loc[code]
                           ['shortName'], code) for code in codePlot]
labLst2 = ['LSTM Rsq=<{}'.format(thR), 'WRTDS Rsq=<{}'.format(
    thR), 'LSTM Rsq>{}'.format(thR), 'WRTDS Rsq>{}'.format(thR)]
fig, ax = figplot.boxPlot(dataBox, widths=0.5, cLst='brcm',
                          label1=labLst1, label2=labLst2,
                          figsize=(12, 4))
fig.show()

# significance test
dfS = pd.DataFrame(index=codePlot, columns=['all', 'static', 'dilution'])
for code in codePlot:
    indC = DFN.varC.index(code)
    ind1 = np.where(rMat[:, indC] <= thR)[0]
    ind2 = np.where(rMat[:, indC] > thR)[0]
    aa, bb = utils.rmNan(
        [matL[ind1, indC, 3], matW[ind1, indC, 3]], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'static'] = p
    aa, bb = utils.rmNan(
        [matL[ind2, indC, 3], matW[ind2, indC, 3]], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'dilution'] = p
    aa, bb = utils.rmNan(
        [matL[:, indC, 3], matW[:, indC, 3]], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'all'] = p
