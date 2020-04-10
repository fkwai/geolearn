from scipy.optimize import curve_fit
from hydroDL.master import basins
from hydroDL.app import waterQuality, relaCQ
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from hydroDL import utils

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fileSiteNoLst = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst')
siteNoLst = pd.read_csv(fileSiteNoLst, header=None, dtype=str)[0].tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist() if siteNo in siteNoLst]
pdfArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoHBN)
unitConv = 0.3048**3*365*24*60*60/1000**2

dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoHBN)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def func(x, a, b):
    return a*1/(x/b+1)


# cal dw
code = '00955'
pMat2 = np.ndarray([len(siteNoHBN), 2])
for k, siteNo in enumerate(siteNoHBN):
    area = pdfArea.loc[siteNo]['DRAIN_SQKM']
    dfC = usgs.readSample(siteNo, codeLst=usgs.codeLst)
    dfQ = usgs.readStreamflow(siteNo)
    df = dfC.join(dfQ)
    t = df.index.values
    q = df['00060_00003'].values/area*unitConv
    c = df[code].values
    ceq, dw, y = relaCQ.kateModel2(q, c)
    pMat2[k, 0] = ceq
    pMat2[k, 1] = dw


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(8, 6))
    axplot.mapPoint(axM, lat, lon, pMat2[:, 1], s=12)
    figP, axP = plt.subplots(2, 1, figsize=(8, 6))
    axP2 = np.array([axP[0], axP[0].twinx(), axP[1]])
    return figM, axM, figP, axP2, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoHBN[iP]
    dfC = usgs.readSample(siteNo, codeLst=usgs.codeLst)
    dfQ = usgs.readStreamflow(siteNo)
    df = dfC.join(dfQ)
    t = df.index.values
    q = df['00060_00003'].values/area*unitConv
    c = df[code].values
    [q, c], ind = utils.rmNan([q, c])
    t = t[ind]
    qAll = dfQ['00060_00003'].values
    qT = dfQ.index.values
    axplot.plotTS(axP[0], qT, qAll, cLst='b', styLst='--')
    axplot.plotTS(axP[1], t, c)
    axP[2].plot(np.log(q), c, 'k*')
    x = 10**np.linspace(np.log10(np.min(q[q > 0])),
                        np.log10(np.max(q[~np.isnan(q)])), 20)
    ceq0 = pMat2[iP, 0]
    dw0 = pMat2[iP, 1]
    y0 = ceq0*1/(x/dw0+1)
    axP[2].plot(np.log(x), y0, 'r-')
    axP[2].set_title('ceq={:.3f},dw={:.3f}'.format(ceq0, dw0))


figplot.clickMap(funcMap, funcPoint)
