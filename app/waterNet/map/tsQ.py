
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
import matplotlib


# bigger model
DF = dbBasin.DataFrameBasin('Q90new')
# tsmap
dfGeo = gageII.readData(siteNoLst=DF.siteNoLst,
                        varLst=['LAT_GAGE', 'LNG_GAGE', 'DRAIN_SQKM'])
lat = dfGeo['LAT_GAGE']
lon = dfGeo['LNG_GAGE']
area = dfGeo['DRAIN_SQKM']

q = DF.q[:, :, 0]
unitC = (0.3048**3)*24*60*60/1000
R = q/area[None, :]*unitC
P = DF.f[:, :, DF.varF.index('pr')]
# E = DF.f[:, :, DF.varF.index('etr')]

# compare water balance
a = np.nanmean(R, axis=0)
b = np.nanmean(P, axis=0)
fig, ax = plt.subplots(1, 1)
axplot.plot121(ax, a, b)
fig.show()


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(10, 4))
    axplot.mapPoint(axM, lat, lon, nash2)
    figP, axP = plt.subplots(2, 1, figsize=(18, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    k = iP
    # axplot.plotTS(axP, dm.t, [yO[:, k], yP1[:, k], yP2[:, k]],
    #               cLst='krb', styLst='---')
    axplot.plotTS(axP[0], dm.t[:indT], [yO[:indT, k], yP[:indT, k]],
                  cLst='krb', styLst='---')
    axplot.plotTS(axP[1], dm.t[indT:], [yO[indT:, k], yP[indT:, k]],
                  cLst='krb', styLst='---')
    axP[0].set_title('basin {} NSE = {}'.format(dm.siteNoLst[iP], nash2[iP]))


figplot.clickMap(funcMap, funcPoint)
