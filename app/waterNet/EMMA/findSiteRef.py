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
usgs.codePdf

codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']
nc = len(codeLst)
indC = [DF.varC.index(code) for code in codeLst]
countC = np.sum(~np.isnan(DF.c[:, :, indC]), axis=0)
countQ = np.sum(~np.isnan(DF.q[:, :, 1]), axis=0)
indS = np.where((countC > 200).all(axis=1) & (countQ > 10000))[0]
siteNoLst = [DF.siteNoLst[ind] for ind in indS]
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM'],
                        siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE']
lon = dfCrd['LNG_GAGE']
count = countC[indS, :].mean(axis=1)
C = DF.c[:, indS, :][:, :, indC]
Q = DF.q[:, indS, 1]/365*1000


def funcM():
    figM = plt.figure()
    gsM = gridspec.GridSpec(1, 1)
    axM = list()
    axM = mapplot.mapPoint(
        figM, gsM[:, :], lat, lon, count, s=16, cb=True)
    gsP = gridspec.GridSpec(nc, 1)
    figP = plt.figure(figsize=[12, 4])
    gsP = gridspec.GridSpec(nc+1, 1)
    axPLst = list()
    for k in range(nc+1):
        axP = figP.add_subplot(gsP[k, 0])
        axPLst.append(axP)
    axP = np.array(axPLst)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = siteNoLst[iP]
    area = dfCrd.loc[siteNo]['DRAIN_SQKM']
    ref = dfCrd.loc[siteNo]['CLASS']
    dataPlot = np.hstack([C[:, iP, :], Q[:, iP, None]])
    labelLst = list()
    for k, code in enumerate(codeLst):
        labelLst.append('{} {}'.format(usgs.codePdf.loc[code]['shortName'],
                                       countC[indS[iP], k]))
    labelLst.append('Q {}'.format(countQ[indS[iP]]))
    axplot.multiTS(axP, DF.t, dataPlot, labelLst=labelLst)
    axP[0].set_title(siteNo)


figM, figP = figplot.clickMap(funcM, funcP)

siteNo = '04193500'
