
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

dataName = 'G200'
DF = dbBasin.DataFrameBasin('G200')
ep = 500
codeLst = usgs.varC
trainSet = 'rmYr5'
testSet = 'pkYr5'

label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, 1000)
corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, 1000)
corrFile1 = os.path.join(outFolder, corrName1)
corrFile2 = os.path.join(outFolder, corrName2)
corrL1 = np.load(corrFile1)
corrL2 = np.load(corrFile2)


# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)
corrL1[matRm] = np.nan
corrL2[matRm] = np.nan


corrLst1 = [corrL1]
corrLst2 = [corrL2]
caseLst = ['reference']

epLst = list()
outLst = list()
for hs in [16, 32, 64, 128, 512]:
    outName = '{}-{}-{}-hs{}'.format(dataName, label, trainSet, hs)
    if hs <= 64:
        epLst.append(100)
    else:
        epLst.append(500)
    caseLst.append('hs-{}'.format(hs))
    outLst.append(outName)

for rho in [180, 750, 1000]:
    outName = '{}-{}-{}-rho{}'.format(dataName, label, trainSet, rho)
    if rho <= 365:
        epLst.append(50)
    else:
        epLst.append(500)
    caseLst.append('rho-{}'.format(rho))
    outLst.append(outName)

for ep in [100, 200, 300, 400]:
    outName = '{}-{}-{}-rho1000'.format(dataName, label, trainSet)
    epLst.append(ep)
    caseLst.append('ep-{}'.format(ep))
    outLst.append(outName)

matObs = DF.c
bQ = np.isnan(DF.q[:, :, 0])
for k, outName in enumerate(outLst):
    obs1 = DF.extractSubset(matObs, trainSet)
    obs2 = DF.extractSubset(matObs, testSet)
    corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
    corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
    print(outName)
    outFolder = basinFull.nameFolder(outName)
    corrFile1 = os.path.join(outFolder, corrName1)
    corrFile2 = os.path.join(outFolder, corrName2)
    yP, ycP = basinFull.testModel(
        outName, DF=DF, testSet='all', ep=epLst[k], reTest=False)
    varY = basinFull.loadMaster(outName)['varY']
    yOut = np.ndarray(yP.shape)
    for k, var in enumerate(varY):
        temp = yP[:, :, k]
        temp[bQ] = np.nan
        yOut[:, :, k] = temp
    pred2 = DF.extractSubset(yOut, testSet)
    corr2 = utils.stat.calCorr(pred2, obs2)
    corrLst2.append(corr2)


# plot
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 10})

figFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
codeStrLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]

matPlot = np.full([len(corrLst2), len(codeLst)], np.nan)
for k, corr in enumerate(corrLst2):
    matPlot[k, :] = np.nanmedian(corr, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
axplot.plotHeatMap(ax, matPlot*100, labLst=[caseLst, codeStrLst])
title = 'Median Testing Correlation [%]'
ax.set_title(title)
plt.tight_layout()
fig.show()
plt.savefig(os.path.join(figFolder, 'heatmap_hyper'))
plt.savefig(os.path.join(figFolder, 'heatmap_hyper.svg'))



# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

# plot trend
var = DF.varC.copy()
var.remove('00400')
iC = np.array([DF.varC.index(code) for code in var])
iR = np.argsort(np.nanmedian(matLR[:, iC], axis=0))
ind = iC[iR]
varP = [DF.varC[k] for k in ind]
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
x = np.nanmedian(matLR[:, ind], axis=0)
for k,case in enumerate(caseLst[:]):    
    y = np.nanmedian(corrLst2[k][:, ind], axis=0)
    if case == 'reference':
        ax.plot(x, y, '-*', label=case,color='k')
    elif case[:2]=='hs':
        hs=int(case.split('-')[1])
        ax.plot(x, y, '-*', label=case,color='r',alpha=min(1,hs/256))
    elif case[:3]=='rho':
        rho=int(case.split('-')[1])
        ax.plot(x, y, '-*', label=case,color='b',alpha=rho/1000)
    elif case[:2]=='ep':
        ep=int(case.split('-')[1])
        ax.plot(x, y, '-*', label=case,color='g',alpha=ep/500)
txtLst = list()

for k, code in enumerate(varP):
    txt = ax.text(x[k], y[k]+0.1, usgs.codePdf.loc[code]['shortName'])
    txtLst.append(txt)
ax.legend()
fig.show()
plt.savefig(os.path.join(figFolder, 'twoDim_hyper'))
plt.savefig(os.path.join(figFolder, 'twoDim_hyper.svg'))