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

DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC

# LSTM
ep = 1000
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
dataName = 'G200'
corrLst1 = list()
corrLst2 = list()
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
matRm = (count1 < 160) & (count2 < 40)
for corr in [corrW1, corrW2] + corrLst1 + corrLst2:
    corr[matRm] = np.nan

# load linear/seasonal
dirPar = os.path.join(kPath.dirWQ, 'modelStat', 'LR-All', 'QS', 'param')
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan


# data quality
ns = len(DF.siteNoLst)
nt=len(DF.t)
nc=len(DF.varC)
DF.c.shape
code = '00600'

iC = DF.varC.index(code)
bC = ~np.isnan(DF.c[:, :, iC])
ddLst=list()
for k in range(ns):
    tC = DF.t[bC[:, k]]
    dt = tC[1:] - tC[:-1]
    dd = dt.astype('timedelta64[D]').astype(int)
    ddLst.append(dd)
ddAll=np.concatenate(ddLst)

fig,axes=plt.subplots(1,3)
axes[0].hist(ddAll,range=[0,30],bins=30)
axes[1].hist(ddAll,range=[0,365],bins=52)
axes[2].hist(ddAll,range=[365,365*10],bins=9)
fig.show()

# find interval after remove gaps
matInt=np.full([ns,nc],np.nan)
matRate=np.full([ns,nc],np.nan)
matA=np.full([ns,nc],np.nan)
matB=np.full([ns,nc],np.nan)
matC=np.full([ns,nc],np.nan)

the=180

ddLst=list()
for iC,code in enumerate(DF.varC):        
    bC = ~np.isnan(DF.c[:, :, iC])
    for k in range(ns):
        tC = DF.t[bC[:, k]]
        if len(tC)>10:
            dt = tC[1:] - tC[:-1]
            dd = dt.astype('timedelta64[D]').astype(int)
            n=np.sum(dd)
            a=np.sum(dd[dd<=30])/n
            b=np.sum(dd[(dd>30)&(dd<365)])/n
            c=1-a-b
            matInt[k,iC]=np.mean(dd[dd<=the])
            matRate[k,iC]=(np.sum(dd)-np.sum(dd[dd>the]))/np.sum(dd)
            matA[k,iC]=a
            matB[k,iC]=b
            matC[k,iC]=c

a=np.sum(dd[dd<=35])
b=np.sum(dd[(dd>35)&(dd<365)])
c=np.sum(dd)-a-b
code='00600'
iC = DF.varC.index(code)
fig,ax=plt.subplots(1,1)
ax.plot(matInt[:,iC],matRate[:,iC],'*')
fig.show()

code='00915'
iC = DF.varC.index(code)
fig,ax=plt.subplots(1,1)
ax.plot(matInt[:,iC],matLR[:,iC],'*')
fig.show()
fig,ax=plt.subplots(1,1)
ax.plot(matRate[:,iC],matLR[:,iC],'*')
fig.show()


# ternary plot
import plotly.express as px

code='00915'
iC = DF.varC.index(code)
fig = px.scatter_ternary(a=matA[:,iC],b=matB[:,iC],c=matC[:,iC],color=matLR[:,iC])
fig.show()

fig = px.scatter_ternary(b=corrW2[:,iC]**2,c=corrL2[:,iC]**2,a=matLR[:,iC])
fig.show()

fig = px.scatter_ternary(a=[0.1],c=[0.1],b=[0.1])
fig.show()

## gageII vs performance with filter
dfG = gageII.readData(siteNoLst=DF.siteNoLst)
dfG = gageII.updateCode(dfG)
lat = dfG['LAT_GAGE'].values
lon = dfG['LNG_GAGE'].values
# remove some attrs
colLst = dfG.columns.tolist()
for yr in range(1950, 2010):
    colLst.remove('PPT{}_AVG'.format(yr))
    colLst.remove('TMP{}_AVG'.format(yr))
for yr in range(1900, 2010):
    colLst.remove('wy{}'.format(yr))
monthLst = ['JAN', 'FEB', 'APR', 'MAY', 'JUN',
            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
for m in monthLst:
    colLst.remove('{}_PPT7100_CM'.format(m))
    colLst.remove('{}_TMP7100_DEGC'.format(m))
x=dfG[colLst].values
y=corrL2

codeLst=DF.varC
precAry=np.array([10,20,30,40,50,60,70,80,90])
# precAry=np.array(range(5,100,5))

corrQS = np.full([len(colLst), len(codeLst)], np.nan)
nx=len(colLst)
for ix,col in enumerate(colLst):
    for iy,code in enumerate(codeLst):
        print(ix,iy)
        print(col,code)
        a=x[:,ix]
        b=y[:,iy]
        kv=np.where(~np.isnan(b))[0]
        ks=np.argsort(b[kv])
        bb=b[kv[ks]]
        aa=a[kv[ks]]
        ind=precAry/100*len(kv)
        ind=np.round(ind).astype(int)
        ap=aa[ind]
        bp=bb[ind]
        corrQS[ix,iy]=np.corrcoef(ap,bp)[0,1]

import matplotlib
indF = np.unique(np.where(np.abs(corrQS) > 0.85)[0])
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
fig, ax = plt.subplots(1, 1,figsize=(18,10))
labLst = [codeLst, [colLst[ind] for ind in indF]]
axplot.plotHeatMap(ax, corrQS[indF, :].T*100, labLst=labLst)
plt.tight_layout()
fig.show()

fig,ax=plt.subplots(1,1)
ax.plot(a,b,'b*')
# ax.plot(ap,bp,'r*')
fig.show()


plt.plot(aa,bb,'*')
plt.show()

ap=np.random.randn(10)
bp=np.random.randn(10)
np.corrcoef(ap,bp)[0,1]