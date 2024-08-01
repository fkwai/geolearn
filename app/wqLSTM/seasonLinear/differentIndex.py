
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
dirParLst = [r'C:\Users\geofk\work\waterQuality\modelStat\LR\QS\param',
          r'C:\Users\geofk\work\waterQuality\modelStat\LR\QST\param',
          r'C:\Users\geofk\work\waterQuality\modelStat\LR-log\QS\param',
          r'C:\Users\geofk\work\waterQuality\modelStat\LR-log\QST\param',
          r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\Q\param',
          r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\S\param',            
]
sLst=['QS','QST','QS-log','QST-log','Q','S']
matLst=list()
for dirPar in dirParLst:
    mat = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirPar, code)
        dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
        mat[:, k] = dfCorr['rsq'].values
    mat[matRm] = np.nan
    matLst.append(mat)


codeStrLst=list()
for code in DF.varC: 
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
figM, axM = figplot.scatter121Batch(
        mat[0], mat[2], np.ones( mat[0].shape), codeStrLst, [5,4], 
        optCb=0,  ticks=[0, 0.5, 1])
figM.show()


# box
codeStrLst = list()
dataPlot = list()
for code in DF.varC: 
    k=DF.varC.index(code)   
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    dataPlot.append([matLst[0][:, k], matLst[1][:, k],
                     matLst[2][:, k],matLst[3][:, k]])
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=codeStrLst)
plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
fig.show()


dfCorr=pd.DataFrame(index=sLst[:4],columns=['lR','wR','lKGE','wKGE','KGE','dR','dKGE'])
# corr between delta and simplicity
codeRm=['00010','00300','00400','00405','00681']
ind = [codeLst.index(code) for code in codeLst if code not in codeRm]
dR=corrL2**2 - corrW2**2
dRmean = np.nanmedian(dR, axis=0)
dK=kgeL2-kgeW2
dKmean = np.nanmedian(dK, axis=0)
lR=np.nanmedian(corrL2, axis=0)
wR=np.nanmedian(corrW2, axis=0)
lKGE=np.nanmedian(kgeL2, axis=0)
wKGE=np.nanmedian(kgeW2, axis=0)


for s0,a0 in zip(sLst[:4],matLst[:4]):
    a = np.nanmedian(a0, axis=0)
    codeRm=['00010','00300','00400','00405','00681']
    ind = [codeLst.index(code) for code in codeLst if code not in codeRm]
    dfCorr.at[s0,'dR']=np.corrcoef(a[ind], dRmean[ind])[0,1]
    codeRm=['00010','00300','00955','80154','00681']
    ind = [codeLst.index(code) for code in codeLst if code not in codeRm]
    dfCorr.at[s0,'dKGE']=np.corrcoef(a[ind], dKmean[ind])[0,1]
    dfCorr.at[s0,'lR']=np.corrcoef(a, lR)[0,1]
    dfCorr.at[s0,'wR']=np.corrcoef(a, wR)[0,1]
    dfCorr.at[s0,'lKGE']=np.corrcoef(a, lKGE)[0,1]
    dfCorr.at[s0,'wKGE']=np.corrcoef(a, wKGE)[0,1]


