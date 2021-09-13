
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
# DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.newC


# LSTM
ep = 500
dataName = 'G200N'
trainSet = 'rmR20'
testSet = 'pkR20'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corr-{}-Ep{}.npy'.format(trainSet, ep)
corrName2 = 'corr-{}-Ep{}.npy'.format(testSet, ep)
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
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 160) & (count2 < 40)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan


dataPlot = list()
codePlot = codeLst
codeStrLst = [usgs.codePdf.loc[code]
              ['shortName'] + '\n'+code for code in codePlot]
thR = 0.3
labLst2 = ['LSTM Rsq=<{}'.format(thR), 'WRTDS Rsq=<{}'.format(thR),
           'LSTM Rsq>{}'.format(thR), 'WRTDS Rsq>{}'.format(thR)]
for code in codePlot:
    ic = codeLst.index(code)
    ind1 = np.where(matLR[:, ic] <= thR)[0]
    ind2 = np.where(matLR[:, ic] > thR)[0]
    dataPlot.append([corrL2[ind1, ic], corrW2[ind1, ic],
                     corrL2[ind2, ic], corrW2[ind2, ic]])
    # dataPlot.append([corrL1[:, ic],corrL2[:, ic], corrW1[:, ic],corrW2[:, ic]])
fig, axes = figplot.boxPlot(dataPlot, widths=0.5, figsize=(12, 4),
                            label1=codeStrLst, label2=labLst2, cLst='rbmc')
fig.show()
dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
# plt.savefig(os.path.join(dirPaper, 'box_weathering'))


# significance test
dfS = pd.DataFrame(index=codePlot, columns=['all', 'static', 'dilution'])
for code in codePlot:
    indC = codeLst.index(code)
    ind1 = np.where(matLR[:, indC] <= thR)[0]
    ind2 = np.where(matLR[:, indC] > thR)[0]
    aa, bb = utils.rmNan(
        [corrL2[ind1, indC], corrW2[ind1, indC]], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'static'] = p
    aa, bb = utils.rmNan(
        [corrL2[ind2, indC], corrW2[ind2, indC]], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'dilution'] = p
    aa, bb = utils.rmNan(
        [corrL2[:, indC], corrW2[:, indC]], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'all'] = p
