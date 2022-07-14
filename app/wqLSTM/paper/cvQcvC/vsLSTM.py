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
trainSet = 'rmYr5'
testSet = 'pkYr5'
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)

matCV = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
q = DF.q[:, :, 1]
cvQ = np.nanstd(q, axis=0)/np.nanmean(q, axis=0)
for k, code in enumerate(codeLst):
    c = DF.c[:, :, k]
    cvC = np.nanstd(c, axis=0)/np.nanmean(c, axis=0)
    matCV[:, k] = cvC/cvQ

# LSTM
ep = 1000
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

dirParLst = [r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param',
             r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\Q\param',
             r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\S\param']
saveNameLst = ['QS', 'Q', 'S']
dictLR = dict()
for dirPar, saveName in zip(dirParLst, saveNameLst):
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirPar, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    dictLR[saveName] = matLR


# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for data in [corrL1, corrL2, corrW1, corrW2, matCV]:
    data[matRm] = np.nan


# plot map
lat, lon = DF.getGeo()
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(5, 4)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 5, 4)
    ax = mapplot.mapPoint(fig, gs[j:j+1, i:i+1], lat, lon,
                          matCV[:, k], cb=True)
    codeStr = usgs.codePdf.loc[code]['shortName']
    ax.set_title('{} {}'.format(code, codeStr))
plt.tight_layout()
fig.show()

# LSTM vs CV
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(5, 4)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 5, 4)
    ax = fig.add_subplot(gs[j:j+1, i:i+1])
    data = corrW2[:, k]**2-corrL2[:, k]**2
    data = dictLR['QS'][:, k]
    ax.plot(data, matCV[:, k], '*')
    codeStr = usgs.codePdf.loc[code]['shortName']
    ax.set_title('{} {}'.format(code, codeStr))
plt.tight_layout()
fig.show()

# paper figure
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
codeGroup = [
    ['00010', '00300'],
    ['00915', '00925', '00930', '00955'],
    ['00600', '00605', '00618', '00660', '00665', '00681', '71846'],
    ['00095', '00400', '00405', '00935', '00940', '00945', '80154']
]
colorGroup = 'rmgb'
labGroup = ['stream', 'weathering', 'nutrient', 'mix']
a0 = matCV
b0 = corrL2**2 - corrW2**2
a = np.nanmean(matCV, axis=0)
b = np.nanmean(corrL2**2 - corrW2**2, axis=0)
c = np.nanmean(corrL2**2, axis=0)
c = np.power(c*10, 3)*2

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
for k in range(len(codeLst)):
    codeStr = usgs.codePdf.loc[codeLst[k]]['shortName']
    if codeStr in usgs.dictLabel.keys():
        ax.text(a[k], b[k], usgs.dictLabel[codeStr], fontsize=16)
    else:
        ax.text(a[k], b[k], codeStr, fontsize=16)
for codeG, colorG, labG in zip(codeGroup, colorGroup, labGroup):
    ind = [codeLst.index(code) for code in codeG]
    ax.scatter(a[ind], b[ind], s=c[ind], color=colorG, label=labG)
    for k in ind:
        aa = [np.nanpercentile(a0[:, k], 25), np.nanpercentile(a0[:, k], 75)]
        bb = [np.nanpercentile(b0[:, k], 25), np.nanpercentile(b0[:, k], 75)]
        ax.plot([a[k], a[k]], bb, color=colorG,
                linestyle='dashed', linewidth=0.5)
        ax.plot(aa, [b[k], b[k]], color=colorG,
                linestyle='dashed', linewidth=0.5)
ax.axhline(0, color='k')
# ax.axvline(0.4, color='k')
ax.set_xlabel('CVc / CVq')
ax.set_ylabel('LSTM Rsq minus WRTDS Rsq')
fig.show()

saveFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
fig.savefig(os.path.join(saveFolder, 'cv2LSTM'))
fig.savefig(os.path.join(saveFolder, 'cv2LSTM.svg'))

# calculate a coefficient
codeCal = codeLst.copy()
codeCal.remove('80154')
ind = [codeLst.index(code) for code in codeCal]
np.corrcoef(a[ind], b[ind])
