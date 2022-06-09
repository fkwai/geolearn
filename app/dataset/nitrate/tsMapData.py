
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
latA, lonA = DF.getGeo()

code = '00618'
ic = DF.varC.index(code)
countC = np.sum(~np.isnan(DF.c[:, :, ic]), axis=0)
indS = np.where(countC >= 200)[0]

C = DF.c[:, indS, ic]
Q = DF.q[:, indS, 1]

lat = latA[indS]
lon = lonA[indS]
siteNoSel = [DF.siteNoLst[x] for x in indS]
matR = np.nanmean(C, axis=0)


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
    siteNo = siteNoSel[iP]
    t=DF.t
    c = C[:, iP]
    q = Q[:, iP]
    axP[0].plot(t, q, '-b')
    axP[1].plot(t, c, '*r')
    axP[0].xaxis_date()
    axP[2].plot(np.log(q), c, 'k*')
    titleStr = '{} {} {:.2f}'.format(code, siteNo, matR[iP])
    axP[0].set_title(titleStr)
    print(titleStr)


figM, figP = figplot.clickMap(funcM, funcP)


col = ['lat', 'lon', 'meanC']
dfMap = pd.DataFrame(index=siteNoSel, columns=col)
dfMap['lat'] = lat
dfMap['lon'] = lon
dfMap['meanC'] = matR

dirMap = r'C:\Users\geofk\work\map\usgs\nitrate'
dfMap.to_csv(os.path.join(dirMap, '{}_mean.csv'.format(code)))