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

code = '00955'
indC = DF.varC.index(code)
countC = np.sum(~np.isnan(DF.c[:, :, indC]), axis=0)
countQ = np.sum(~np.isnan(DF.q[:, :, 1]), axis=0)
indS = np.where((countC > 200) & (countQ > 10000))[0]
siteNoLst = [DF.siteNoLst[ind] for ind in indS]
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM'],
                        siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE']
lon = dfCrd['LNG_GAGE']

t = DF.t
C = DF.c[:, indS, indC]
Q = DF.q[:, indS, 1]/365*1000
matR = utils.stat.calCorr(C, Q)**2


def funcM():
    figM = plt.figure()
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, matR, s=16, cb=True)
    gsP = gridspec.GridSpec(1, 3)
    figP = plt.figure(figsize=[12, 4])
    gsP = gridspec.GridSpec(1, 3)
    axP1 = figP.add_subplot(gsP[0, :2])
    axP2 = axP1.twinx()
    axP3 = figP.add_subplot(gsP[0, 2])
    axP = np.array([axP1, axP2, axP3])
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = siteNoLst[iP]
    area = dfCrd.loc[siteNo]['DRAIN_SQKM']
    ref = dfCrd.loc[siteNo]['CLASS']
    c = C[:, iP]
    q = Q[:, iP]
    axP[0].plot(t, q, '-b')
    axP[1].plot(t, c, '*r')
    axP[0].xaxis_date()
    axP[2].plot(np.log(q), c, 'k*')
    titleStr = '{} {} {} {:.2f}'.format(code, siteNo, ref, matR[iP])
    axP[0].set_title(titleStr)
    print(titleStr)


figM, figP = figplot.clickMap(funcM, funcP)
