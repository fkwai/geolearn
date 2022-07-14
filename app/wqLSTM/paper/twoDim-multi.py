
import matplotlib.gridspec as gridspec
import matplotlib
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
ep = 500
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
codeGroup = [['00400','00405', '00600', '00605', '00618', '00660',
             '00665', '00681', '71846', '80154'],
             ['00095', '00915', '00925', '00930',
             '00935', '00940', '00945', '00955'],
             ['00010', '00300']]
xLmLst = [[0, 0.3], [0.28, 0.58], [0.6, 0.9]]
yLmLst = [[0, 0.5], [0.28, 0.78], [0.6, 1]]

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 5})

fig = plt.figure(figsize=(14, 3))
gsM = gridspec.GridSpec(1, 5)
ax1 = fig.add_subplot(gsM[0, :2])
ax2 = fig.add_subplot(gsM[0, 2:4])
ax3 = fig.add_subplot(gsM[0, 4])
axes = [ax1, ax2, ax3]

x = np.nanmedian(matLR, axis=0)
txtLst = list()
for k, group in enumerate(codeGroup):
    for code in group:
        ic = codeLst.index(code)
        txt = axes[k].text(x[ic], y[ic], usgs.codePdf.loc[code]['shortName'])
        txtLst.append(txt)
    ic = np.array([codeLst.index(code) for code in group])
    ind = np.argsort(x[ic])
    xx = x[ic[ind]]
    y1 = np.nanmedian(corrW2[:, ic]**2, axis=0)[ind]
    y2 = np.nanmedian(corrLst2[0][:, ic]**2, axis=0)[ind]
    y3 = np.nanmedian(corrLst2[1][:, ic]**2, axis=0)[ind]
    axes[k].plot(xx, y2, 'r*-')
    axes[k].plot(xx, y3, 'm*-')
    axes[k].plot(xx, y1, 'b*-')
    axes[k].plot([xx[0], xx[-1]], [xx[0], xx[-1]], 'k-')
    axes[k].set_xlim(xLmLst[k])
    axes[k].set_ylim(yLmLst[k])
    xx
fig.show()

# adjustText.adjust_text(txtLst)
ic1 = np.array([codeLst.index(code) for code in var1])
ic2 = np.array([codeLst.index(code) for code in var2])

ind1 = np.argsort(x[ic1])
ind2 = np.argsort(x[ic2])
for data, color in zip(dataLst, 'kbr'):
    y = np.nanmean(data**2, axis=0)
    ax.plot(x[ic1[ind1]], y[ic1[ind1]], ls='-', marker='*', color=color)
    ax.plot(x[ic2[ind2]], y[ic2[ind2]],  ls='-', marker='*', color=color)
ax.plot([0, 1], [0, 1], 'k-')

fig.show()
