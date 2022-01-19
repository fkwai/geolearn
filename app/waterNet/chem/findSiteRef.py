import random
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec


DF = dbBasin.DataFrameBasin('G200')

codeLst = ['00618', '00915', '00945', '00955']
nc = len(codeLst)
indC = [DF.varC.index(code) for code in codeLst]
countC = np.sum(~np.isnan(DF.c[:, :, indC]), axis=0)
countQ = np.sum(~np.isnan(DF.q[:, :, 1]), axis=0)
indS = np.where((countC > 200).all(axis=1) & (countQ > 10000))[0]
siteNoLst = [DF.siteNoLst[ind] for ind in indS]
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM'],
                        siteNoLst=siteNoLst)
indRef = np.where(dfCrd['CLASS'] == 'Ref')[0]
lat = dfCrd['LAT_GAGE'][indRef]
lon = dfCrd['LNG_GAGE'][indRef]
siteNoLst = [siteNoLst[ind] for ind in indRef]
t = DF.t
C = DF.c[:, indS[indRef], :][:, :, indC]
Q = DF.q[:, indS[indRef], 1]/365*1000
rLst = list()
for k in range(nc):
    matR = utils.stat.calCorr(C[:, :, 0], Q)**2
    rLst.append(matR)


def funcM():
    figM = plt.figure()
    gsM = gridspec.GridSpec(nc, 1)
    axM = list()
    for k in range(nc):
        axM = mapplot.mapPoint(
            figM, gsM[k, 0], lat, lon, rLst[k], s=16, cb=True)
    gsP = gridspec.GridSpec(nc, 3)
    figP = plt.figure(figsize=[12, 4])
    gsP = gridspec.GridSpec(nc, 3)
    axPLst = list()
    for k in range(nc):
        axP1 = figP.add_subplot(gsP[k, :2])
        axP2 = axP1.twinx()
        axP3 = figP.add_subplot(gsP[k, 2])
        axPLst.append([axP1, axP2, axP3])
    axP = np.array(axPLst)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = siteNoLst[iP]
    area = dfCrd.loc[siteNo]['DRAIN_SQKM']
    ref = dfCrd.loc[siteNo]['CLASS']
    for k in range(nc):
        c = C[:, iP, k]
        q = Q[:, iP]
        axP[k, 0].plot(t, q, '-b')
        axP[k, 1].plot(t, c, '*r')
        axP[k, 0].xaxis_date()
        axP[k, 2].plot(np.log(q), c, 'k*')
        titleStr = '{} {} {} {:.2f}'.format(codeLst[k], siteNo, ref, matR[iP])
        axP[k, 0].set_title(titleStr)
    print(titleStr)


figM, figP = figplot.clickMap(funcM, funcP)
