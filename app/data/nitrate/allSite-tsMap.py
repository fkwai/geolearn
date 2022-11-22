
from cProfile import label
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

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

codeLst = ['00400', '00600', '00605', '00618', '71846']
# DF = dbBasin.DataFrameBasin.new(
#     'allN', siteNoLstAll, varC=codeLst, varF=[], varQ=usgs.varQ, varG=gageII.varLst)
DF = dbBasin.DataFrameBasin('allN')
latA, lonA = DF.getGeo()

# tsMap
code = '00600'
ic = DF.varC.index(code)
countC = np.sum(~np.isnan(DF.c[:, :, ic]), axis=0)
indS = np.where(countC >= 50)[0]

C = DF.c[:, indS, ic]
Q = DF.q[:, indS, 1]

lat = latA[indS]
lon = lonA[indS]
siteNoSel = [DF.siteNoLst[x] for x in indS]
matR = np.nanmedian(C, axis=0)
cs = usgs.codePdf.loc[code]['shortName']


def funcM():
    figM = plt.figure(figsize=(8, 4))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, matR, s=16, cb=True)
    axM.set_title('median {} [mg/l]'.format(cs))
    gsP = gridspec.GridSpec(1, 3)
    figP = plt.figure(figsize=[15, 4])
    gsP = gridspec.GridSpec(1, 3)
    axP1 = figP.add_subplot(gsP[0, :2])
    axP2 = axP1.twinx()
    axP3 = figP.add_subplot(gsP[0, 2])
    axP = np.array([axP1, axP2, axP3])
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = siteNoSel[iP]
    t = DF.t
    c = C[:, iP]
    q = Q[:, iP]
    axP[0].plot(t, q, '-b', label='runoff [mm]')
    axP[1].plot(t, c, '*r', label='{} [mg/l]'.format(cs))
    axP[0].xaxis_date()
    axP[2].plot(np.log(q), c, 'k*')
    titleStr = '{} {} {:.2f}'.format(code, siteNo, matR[iP])
    axP[0].set_title(titleStr)
    axP[0].legend(loc='upper left')
    axP[1].legend(loc='upper right')


figM, figP = figplot.clickMap(funcM, funcP)
