
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
DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC


dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=1000)
codeLst = usgs.varC


# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation
matNan = np.isnan(yP) | np.isnan(yW)
yP[matNan] = np.nan
yW[matNan] = np.nan
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
yP1 = DF.extractSubset(yP, trainSet)
yP2 = DF.extractSubset(yP, testSet)
yW1 = DF.extractSubset(yW, trainSet)
yW2 = DF.extractSubset(yW, testSet)
corrL1 = utils.stat.calCorr(yP1, obs1)
corrL2 = utils.stat.calCorr(yP2, obs2)
corrW1 = utils.stat.calCorr(yW1, obs1)
corrW2 = utils.stat.calCorr(yW2, obs2)
nashL1 = utils.stat.calNash(yP1, obs1)
nashL2 = utils.stat.calNash(yP2, obs2)
nashW1 = utils.stat.calNash(yW1, obs1)
nashW2 = utils.stat.calNash(yW2, obs2)
kgeL1 = utils.stat.calKGE(yP1, obs1)
kgeL2 = utils.stat.calKGE(yP2, obs2)
kgeW1 = utils.stat.calKGE(yW1, obs1)
kgeW2 = utils.stat.calKGE(yW2, obs2)

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan
for kge in [kgeL1, kgeL2, kgeW1, kgeW2]:
    kge[matRm] = np.nan
    kge[np.isinf(kge)] = np.nan
    kge[kge < -1] = np.nan
for nash in [nashL1, nashL2, nashW1, nashW2]:
    nash[matRm] = np.nan
    

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-log\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

codeGroup = [
    ['00010', '00300'],
    ['00915', '00925', '00930', '00955'],
    ['00600', '00605', '00618', '00660', '00665', '00681', '71846'],
    ['00095', '00400', '00405', '00935', '00940', '00945', '80154']
]
# colorGroup = 'rmgb'
colorGroup = ['#e41a1c','#984ea3','#4daf4a','#377eb8']
labGroup = ['stream', 'weathering', 'nutrient', 'mix']
#
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
# matplotlib.rcParams.update({'svg.fonttype': 'none'})


# corr and kge
a0=matLR
b0Lst= [corrL2**2 - corrW2**2,
        kgeL2-kgeW2]
bLst=[np.nanmean(corrL2**2 - corrW2**2, axis=0),
      np.nanmedian(kgeL2-kgeW2, axis=0)]
      

cLst=[np.nanmedian(corrL2,axis=0)**4*500,
         np.nanmedian(kgeL2,axis=0)**2*500]
a = np.nanmean(a0, axis=0)
ylimLst=[[-0.1,0.15],[-0.1,0.15]]
strLst=['corr','kge']

for b0,b,c,ylim,saveStr in zip(b0Lst,bLst,cLst,ylimLst,strLst):    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for k in range(len(codeLst)):
        codeStr = usgs.codePdf.loc[codeLst[k]]['shortName']
        if codeStr in usgs.dictLabel.keys():
            ax.text(a[k], b[k], usgs.dictLabel[codeStr], fontsize=12)
        else:
            ax.text(a[k], b[k], codeStr, fontsize=12)
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
        ax.set_ylim(ylim)
    ax.axhline(0, color='k')
    ax.axvline(0.33, color='k')
    ax.set_xlabel('simplicity')
    # ax.set_ylabel('LSTM Rsq minus WRTDS Rsq')
    # ax.set_ylabel('LSTM KGE minus WRTDS KGE')
    fig.show()
    dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
    plt.savefig(os.path.join(dirPaper, 'fourDim-{}-log'.format(saveStr)))
    plt.savefig(os.path.join(dirPaper, 'fourDim-{}-log.svg'.format(saveStr)))

# plot legend
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for colorG, labG in zip(colorGroup, labGroup):
    ind = [codeLst.index(code) for code in codeG]
    ax.scatter(0, 0, s=100, color=colorG, label=labG)
ax.legend()
ax.set_xlabel('simplicity')
ax.set_ylabel('LSTM Rsq minus WRTDS Rsq')
fig.show()
plt.savefig(os.path.join(dirPaper, 'fourDim_leg'))
plt.savefig(os.path.join(dirPaper, 'fourDim_leg.svg'))

# calculate a coefficient
codeCal = [
    '00915', '00925', '00930', '00955', '00600',
    '00605', '00618', '00660', '00665', '00681', '71846',
    '00095', '00935', '00940', '00945', '80154']
ind = [codeLst.index(code) for code in codeCal]
np.corrcoef(a[ind], b[ind])

# corr between delta and simplicity
codeRm=['00010','00300','00400','00405','00681']
ind = [codeLst.index(code) for code in codeLst if code not in codeRm]
a0=matLR
a = np.nanmedian(a0, axis=0)
b0=corrL2**2 - corrW2**2
b = np.nanmedian(b0, axis=0)
np.corrcoef(a[ind], b[ind])

codeRm=['00010','00300','00955','80154','00681']
ind = [codeLst.index(code) for code in codeLst if code not in codeRm]
a0=matLR
a = np.nanmedian(a0, axis=0)
b0=kgeL2-kgeW2
b = np.nanmedian(b0, axis=0)
np.corrcoef(a[ind], b[ind])

fig,ax=plt.subplots(1,1,figsize=(6,6))
ax.plot(a,b,'*')
ax.plot(a[ind], b[ind],'r*')
fig.show()

b0Lst= [corrL2**2 - corrW2**2,
        kgeL2-kgeW2]
cLst=[np.nanmedian(corrL2,axis=0)**2*500,
         np.nanmedian(kgeL2,axis=0)**2*500]
a = np.nanmedian(a0, axis=0)