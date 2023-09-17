
import scipy.stats as stats
import scipy.stats
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
from sklearn.metrics import r2_score

codeLst = usgs.varC


# LSTM
DF = dbBasin.DataFrameBasin('G200')

ep = 1000
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
# label = 'FPRT2QC'

outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQF-{}-Ep{}.npy'.format(trainSet, ep)
corrName2 = 'corrQF-{}-Ep{}.npy'.format(testSet, ep)
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
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
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
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

fig, axes = plt.subplots(5, 4)
for code in codeLst:
    ic = codeLst.index(code)
    ix, iy = utils.index2d(ic, 5, 4)
    axes[ix, iy].plot(count1[:, ic], corrL1[:, ic], '*')
fig.suptitle(code)
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(count1, corrL2, '*')
axes[1].plot(count2, corrL2, '*')
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(count1, corrW2, '*')
axes[1].plot(count2, corrW2, '*')
fig.show()


fig, axes = plt.subplots(2, 1)
axes[0].plot(count1, corrL2**2-corrW2**2, '*')
axes[1].plot(count2, corrL2**2-corrW2**2, '*')
axes[0].axhline(0)
fig.show()

code = '00915'
ic = codeLst.index(code)
fig, axes = plt.subplots(2, 1)
axes[0].plot(count1[:, ic], corrL2[:, ic], '*')
axes[1].plot(count2[:, ic], corrL2[:, ic], '*')
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(np.nanmean(count1, axis=0), np.nanmean(corrL2, axis=0), '*')
axes[1].plot(np.nanmean(count2, axis=0), np.nanmean(corrL2, axis=0), '*')
fig.show()

# groups
codeGroup = [
    ['00010', '00300'],
    ['00915', '00925', '00930', '00955'],
    ['00600', '00605', '00618', '00660', '00665', '00681', '71846'],
    ['00095', '00400', '00405', '00935', '00940', '00945', '80154']
]
labGroup = ['stream', 'weathering', 'nutrient', 'mix']

fig, axes = plt.subplots(2, 2)
for k, (codeG,  labG) in enumerate(zip(codeGroup, labGroup)):
    iy, ix = utils.index2d(k, 2, 2)
    for kk, code in enumerate(codeG):
        ic = codeLst.index(code)
        codeStr = usgs.codePdf.loc[code]['shortName']
        if codeStr in usgs.dictLabel.keys():
            codeStr = usgs.dictLabel[codeStr]
        v = corrL2[:, ic].copy()
        c = count1[:, ic].copy()
        vv, cc = utils.rmNan([v, c], returnInd=False)
        indSort = np.argsort(cc)
        xLst = list()
        pLst = list()
        n=50
        for i in range(0, len(indSort), n):            
            y = vv[indSort[i:i+n]]
            x = cc[indSort[i:i+n]]
            pLst.append(np.percentile(y, 50))
            xLst.append(np.mean(x))
        # c[c > 1500] = 1501
        axes[iy, ix].plot(cc, vv, '*', color='C'+str(kk), label=codeStr)
        # axes[ix, iy].plot(xLst, pLst, '-', color='C'+str(kk))
    axplot.titleInner(axes[ix, iy], labG, top=False)
    axes[iy, ix].legend()
    # axes[iy, ix].set_xlim([0, 1501])
fig.show()


# count of training vs performance
n = 30
fig, axes = plt.subplots(4, 5)
for kk, code in enumerate(codeLst):
    ic = codeLst.index(code)
    ix, iy = utils.index2d(ic, 4, 5)
    # axes[ix, iy].set_title(code)
    axes[ix, iy].set_xlim([0, 1501])
    v = corrW2[:, ic]
    c = count1[:, ic]
    axes[ix, iy].plot(c, v, 'k.')
    vv, cc = utils.rmNan([v, c], returnInd=False)
    coefficients = np.polyfit(cc, vv, 1)
    poly = np.poly1d(coefficients)
    trend_y = poly(cc)
    axes[ix, iy].plot(cc, trend_y, color='yellow')
    indSort = np.argsort(cc)
    p1Lst = list()
    p2Lst = list()
    p3Lst = list()
    xLst = list()
    for k in range(0, len(indSort), n):
        k
        y = vv[indSort[k:k+n]]
        x = cc[indSort[k:k+n]]
        p1Lst.append(np.percentile(y, 25))
        p2Lst.append(np.percentile(y, 50))
        p3Lst.append(np.percentile(y, 75))
        xLst.append(np.mean(x))
    axes[ix, iy].plot(xLst, p1Lst, 'b-*')
    axes[ix, iy].plot(xLst, p2Lst, 'r-*')
    axes[ix, iy].plot(xLst, p3Lst, 'g-*')
    codeStr = usgs.codePdf.loc[code]['shortName']
    if codeStr in usgs.dictLabel.keys():
        codeStr = usgs.dictLabel[codeStr]
    axplot.titleInner(axes[ix, iy], '{}'.format(codeStr), top=False)
    # axes[ix, iy].set_xlim([0, 500])
fig.show()

# load simplicity index
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan


# count of training vs performance
n = 30
fig, axes = plt.subplots(4, 5)
for kk, code in enumerate(codeLst):
    ic = codeLst.index(code)
    ix, iy = utils.index2d(ic, 4, 5)
    # axes[ix, iy].set_title(code)
    axes[ix, iy].set_xlim([0, 1501])
    v = matLR[:, ic]
    c = count1[:, ic]
    axes[ix, iy].plot(c, v, 'k.')
    vv, cc = utils.rmNan([v, c], returnInd=False)
    coefficients = np.polyfit(cc, vv, 1)
    poly = np.poly1d(coefficients)
    trend_y = poly(cc)
    axes[ix, iy].plot(cc, trend_y, color='yellow')
    indSort = np.argsort(cc)
    p1Lst = list()
    p2Lst = list()
    p3Lst = list()
    xLst = list()
    for k in range(0, len(indSort), n):
        k
        y = vv[indSort[k:k+n]]
        x = cc[indSort[k:k+n]]
        p1Lst.append(np.percentile(y, 25))
        p2Lst.append(np.percentile(y, 50))
        p3Lst.append(np.percentile(y, 75))
        xLst.append(np.mean(x))
    axes[ix, iy].plot(xLst, p1Lst, 'b-*')
    axes[ix, iy].plot(xLst, p2Lst, 'r-*')
    axes[ix, iy].plot(xLst, p3Lst, 'g-*')
    codeStr = usgs.codePdf.loc[code]['shortName']
    if codeStr in usgs.dictLabel.keys():
        codeStr = usgs.dictLabel[codeStr]
    axplot.titleInner(axes[ix, iy], '{}'.format(codeStr), top=False)
    # axes[ix, iy].set_xlim([0, 500])
fig.show()


# simplicity vs performance
n = 30
fig, axes = plt.subplots(4, 5)
for kk, code in enumerate(codeLst):
    ic = codeLst.index(code)
    ix, iy = utils.index2d(ic, 4, 5)
    x = matLR[:, ic]
    y = corrL2[:, ic]
    axes[ix, iy].plot(x, y, 'k.')
    xx, yy = utils.rmNan([x, y], returnInd=False)
    coefficients = np.polyfit(xx, yy, 1)
    poly = np.poly1d(coefficients)
    trend_y = poly(xx)
    r2= np.corrcoef(xx,yy)[0,1]**2
    axes[ix, iy].plot(xx, trend_y, color='yellow')
    codeStr = usgs.codePdf.loc[code]['shortName']
    if codeStr in usgs.dictLabel.keys():
        codeStr = usgs.dictLabel[codeStr]
    axplot.titleInner(axes[ix, iy], '{} {:.2f}'.format(codeStr, r2), top=False)
fig.show()


# count vs performance
n = 30
fig, axes = plt.subplots(4, 5)
for kk, code in enumerate(codeLst):
    ic = codeLst.index(code)
    ix, iy = utils.index2d(ic, 4, 5)
    x = np.log(count1[:, ic])
    y = corrL2[:, ic]
    axes[ix, iy].plot(x, y, 'k.')
    xx, yy = utils.rmNan([x, y], returnInd=False)
    r2= r2_score(xx, yy)
    coefficients = np.polyfit(xx, yy, 1)
    poly = np.poly1d(coefficients)
    trend_y = poly(xx)
    r2= np.corrcoef(xx,yy)[0,1]**2
    axes[ix, iy].plot(xx, trend_y, color='yellow')
    codeStr = usgs.codePdf.loc[code]['shortName']
    if codeStr in usgs.dictLabel.keys():
        codeStr = usgs.dictLabel[codeStr]
    axplot.titleInner(axes[ix, iy], '{} {:.2f}'.format(codeStr, r2), top=False)
fig.show()
