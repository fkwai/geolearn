
import matplotlib.dates as mdates
import random
import scipy
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib
import matplotlib.gridspec as gridspec


DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC


# LSTM corr
ep = 1000
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

# load TS
DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=1000)
codeLst = usgs.varC
# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation
yPC=yP.copy()
yWC=yW.copy()
matNan = np.isnan(yP) | np.isnan(yW)
yPC[matNan] = np.nan
yWC[matNan] = np.nan
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
yP1 = DF.extractSubset(yPC, trainSet)
yP2 = DF.extractSubset(yPC, testSet)
yW1 = DF.extractSubset(yWC, trainSet)
yW2 = DF.extractSubset(yWC, testSet)
importlib.reload(utils.stat)

statStr='Corr'
func=getattr(utils.stat,'cal'+statStr)
statL1=func(yP1,obs1)
statL2=func(yP2,obs2)
statW1=func(yW1,obs1)
statW2=func(yW2,obs2)

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)
for stat in [statL1, statL2, statW1, statW2]:
    stat[matRm] = np.nan


# load linear/seasonal
dirPar = os.path.join(kPath.dirWQ,'modelStat','LR-All','QS','param')
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

# ts map
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})

lat, lon = DF.getGeo()
code = '00010'
indC = codeLst.index(code)
indS = np.where(~matRm[:, indC])[0]
importlib.reload(figplot)
importlib.reload(axplot)
yrLst = np.arange(1985, 2020, 5).tolist()
ny = len(yrLst)

xMat = np.ndarray([len(indS), 3])
yMat = np.ndarray([len(indS), 3])
xMat[:, 0] = statL2[indS, indC]
yMat[:, 0] = statW2[indS, indC]
xMat[:, 1] = lon[indS]
yMat[:, 1] = lat[indS]
xMat[:, 2] = lon[indS]
yMat[:, 2] = lat[indS]


# calculate count
a = matLR[indS, indC]
b = statW2[indS, indC]
c = statL2[indS, indC]
d1=count1[indS,indC]
d2=count2[indS,indC]
fig,ax=plt.subplots(1,1)
cs = axplot.scatter121(ax, b,c,d1,vR=[0,500])
fig.colorbar(cs, orientation='vertical')
fig.show()
fig,ax=plt.subplots(1,1)
ax.plot(d1,d2,'*')
fig.show()


ind1 = np.where((d1<np.percentile(d1,60)) & (d1>np.percentile(d1,40)))[0]
ind2 = np.where((d2<np.percentile(d2,60)) & (d2>np.percentile(d2,40)))[0]
ind=np.intersect1d(ind1, ind2)
# ind=ind1
a[ind]
b[ind]
c[ind]
indS2=indS[ind]
d1min=np.percentile(d1,0)
d1max=np.percentile(d1,90)
# another ts pick

import importlib
importlib.reload(axplot)
# value to rank percentile
xMatRank=xMat.copy()
yMatRank=yMat.copy()
xMatRank[:,0]=np.argsort(np.argsort(xMatRank[:,0]))/xMatRank.shape[0]
yMatRank[:,0]=np.argsort(np.argsort(yMatRank[:,0]))/yMatRank.shape[0]

# dfData=np.stack([xMat[:,0],xMatRank[:,0],yMat[:,0],yMatRank[:,0]]).T
# df=pd.DataFrame(data=dfData)
# df.to_csv('temp')
xMat2=xMatRank[ind, :]
yMat2=yMatRank[ind, :]


def funcM():
    figM = plt.figure(figsize=(12, 6))
    gsM = gridspec.GridSpec(2, 2)
    labelLst = ['scatter', 'map1', 'map2']
    axS = figM.add_subplot(gsM[:, 0])
    axS.set_label(labelLst[0])
    # axS.plot(xMat[:,0],yMat[:,0],'.',color='grey',alpha=0.5)
    cs0 = axplot.scatter121(axS, xMatRank[:,0], yMatRank[:,0], d1,vR=[d1min,d1max],alpha=0.3)
    cs = axplot.scatter121(axS, xMat2[:,0], yMat2[:,0], d1[ind],vR=[d1min,d1max],edgecolors='black')
    plt.colorbar(cs, orientation='vertical')

    axM1 = mapplot.mapPoint(
        figM, gsM[0, 1:], xMat2[:,1], yMat2[:,1], statL2[indS2, indC])
    axM1.set_label(labelLst[1])
    axM2 = mapplot.mapPoint(
        figM, gsM[1, 1:], xMat2[:,2], yMat2[:,2],
        statL2[indS2, indC]**2-statW2[indS2, indC]**2)
    axM2.set_label(labelLst[2])
    axM = np.array([axS, axM1, axM2])
    figP = plt.figure(figsize=(15, 3))
    gsP = gridspec.GridSpec(1, ny, wspace=0)
    axP0 = figP.add_subplot(gsP[0, 0])
    axPLst = [axP0]
    for k in range(1, ny):
        axP = figP.add_subplot(gsP[0, k], sharey=axP0)
        axPLst.append(axP)
    axP = np.array(axPLst)
    return figM, axM, figP, axP, xMat2, yMat2, labelLst


def funcP(axP, iP, iM):
    print(iP, iM)
    k = indS2[iP]
    dataPlot = [yW[:, k, indC], yP[:, k, indC],
                DF.c[:, k, DF.varC.index(code)]]    
    cLst = 'kbr'
    legLst = ['WRTDS', 'LSTM', 'Obs']
    axplot.multiYrTS(axP,  yrLst, DF.t, dataPlot, cLst=cLst, legLst=legLst)
    titleStr = '{} {:.2f} {:.2f}'.format(
        DF.siteNoLst[k], statL2[k, indC], statW2[k, indC])
    print(titleStr)


figM, figP = figplot.clickMulti(funcM, funcP)


ind1 = np.where(b > c)[0]
ind2 = np.where(b < c)[0]
np.median(a[ind1])
np.median(a[ind2])
len(ind1)
len(ind2)

the = np.percentile(a, 40)
ind1 = np.where(a < the)[0]
ind2 = np.where(a > the)[0]
np.median(b[ind1])
np.median(c[ind1])
np.median(b[ind2])
np.median(c[ind2])
np.nanmedian(matLR, axis=0)

np.median(b)
np.median(c)
np.mean(b)
np.mean(c)

# for a specfic site
# code = '00665'
# siteNo = '01631000'
# indS = DF.siteNoLst.index(siteNo)
# # extract 5 yr
# indLst = list()
# ny = len(yrLst)
# ty = DF.t.astype('M8[Y]').astype(str).astype(int)
# for yr in yrLst:
#     bp = np.in1d(ty, yr)
#     ind = np.where(bp)[0]
#     indLst.append(ind)
# indAry = np.concatenate(indLst)
# a = yW[:, indS, indC]
# b = yP[:, indS, indC]
# c = DF.c[:, indS, indC]
# aY = a[indAry]
# bY = b[indAry]
# cY = c[indAry]
# [aa, bb, cc], indT = utils.rmNan([aY, bY, cY])
# np.corrcoef(aa, cc)
# np.corrcoef(bb, cc)
# fig, ax = plt.subplots(1, 1)
# ax.plot(cc, aa, 'b*')
# ax.plot(cc, bb, 'r*')
# fig.show()
# fig, ax = plt.subplots(1, 1)
# ax.plot(cc, aa, 'b*')
# ax.plot(cc, bb, 'r*')
# fig.show()
