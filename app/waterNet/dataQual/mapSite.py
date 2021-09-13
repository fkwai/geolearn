from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

DF = dbBasin.DataFrameBasin('G400')

siteNoLst = DF.siteNoLst
codeLst = DF.varC
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM'],
                        siteNoLst=siteNoLst)

matCount = np.zeros([len(siteNoLst), len(codeLst)])
for ic, code in enumerate(codeLst):
    temp = ~np.isnan(DF.q[:, :, 0]) & ~np.isnan(DF.c[:, :, ic])
    matCount[:, ic] = np.sum(temp, axis=0)


code = '00915'
indC = codeLst.index(code)
lat = dfCrd.loc[siteNoLst]['LAT_GAGE']
lon = dfCrd.loc[siteNoLst]['LNG_GAGE']
t = DF.t


def funcM():
    figM, axM = plt.subplots(1, 1, figsize=(6, 4))
    axplot.mapPoint(axM, lat, lon, matCount[:, indC], s=16, cb=True)
    figP, axP1 = plt.subplots(1, 1, figsize=(12, 4))
    axP2 = axP1.twinx()
    axP = np.array([axP1, axP2])
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    siteNo = siteNoLst[iP]
    area = dfCrd.loc[siteNo]['DRAIN_SQKM']
    c = DF.c[:, iP, indC]
    q = DF.q[:, iP, 0]
    axP[1].plot(t, q, '-b')
    axP[0].plot(t, c, '*r')
    axP[0].xaxis_date()
    titleStr = '{} {} {} {}'.format(code, siteNo, area, matCount[iP, indC])
    axP[0].set_title(titleStr)
    print(titleStr)


figM, figP = figplot.clickMap(funcM, funcP)
