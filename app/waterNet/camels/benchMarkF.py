
import matplotlib.gridspec as gridspec
import json
import numpy as np
from numpy import double
import pandas as pd
import os
import hydroDL
from hydroDL.data import camels, usgs, dbBasin
import importlib
import matplotlib.pyplot as plt
from hydroDL.post import axplot, mapplot
from hydroDL.utils.time import t2dt
from hydroDL.master import basinFull, slurm
from hydroDL import utils


df = dbBasin.DataFrameBasin('camelsK')
dfN = dbBasin.DataFrameBasin('camelsN')
dfD = dbBasin.DataFrameBasin('camelsD')
dfM = dbBasin.DataFrameBasin('camelsM')


siteNoLst = df.siteNoLst

df.saveSubset('benchmark', sd='1980-01-01',
              ed='2009-12-31', siteNoLst=siteNoLst)
dfN.saveSubset('benchmark', sd='1980-01-01',
               ed='2009-12-31', siteNoLst=siteNoLst)
dfD.saveSubset('benchmark', sd='1980-01-01',
               ed='2009-12-31', siteNoLst=siteNoLst)
dfM.saveSubset('benchmark', sd='1980-01-01',
               ed='2009-12-31', siteNoLst=siteNoLst)


k = 0
siteNo = siteNoLst[k]
k1 = dfN.siteNoLst.index(siteNo)
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(df.t, df.f[:, k, 0], '-k', label='ours')
ax.plot(dfN.t, dfN.f[:, k1, 1], '*r', label='NLDAS')
ax.plot(dfD.t, dfD.f[:, k1, 1], '*g', label='DayMet')
ax.plot(dfM.t, dfM.f[:, k1, 1], '*b', label='Maurer')
ax.legend()
ax.set_title(siteNoLst[k])
ax.set_xlim([t2dt(20000101), t2dt(20010101)])
# ax.set_ylim([0,20])
# ax.set_yscale('log')
fig.show()

# map of difference
v = df.extractSubset(df.f[:, :, 0:1], subsetName='benchmark')
vN = dfN.extractSubset(dfN.f[:, :, 1:2], subsetName='benchmark')
vD = dfD.extractSubset(dfD.f[:, :, 1:2], subsetName='benchmark')
vM = dfM.extractSubset(dfM.f[:, :, 1:2], subsetName='benchmark')


corrN = utils.stat.calCorr(v, vN)
corrM = utils.stat.calCorr(v, vM)
corrD = utils.stat.calCorr(v, vD)

corrNM = utils.stat.calCorr(vN, vM)
corrMD = utils.stat.calCorr(vD, vM)
corrND = utils.stat.calCorr(vN, vD)

lat, lon = df.getGeo()
figM = plt.figure()
gsM = gridspec.GridSpec(3, 2)
axM = mapplot.mapPoint(
    figM, gsM[0, 0], lat, lon, corrN, s=16, cb=True, vRange=[0.5, 1])
axM.set_title('gridMet vs NLDAS')
axM = mapplot.mapPoint(
    figM, gsM[1, 0], lat, lon, corrM, s=16, cb=True, vRange=[0.5, 1])
axM.set_title('gridMet vs Maurer')
axM = mapplot.mapPoint(
    figM, gsM[2, 0], lat, lon, corrD, s=16, cb=True, vRange=[0.5, 1])
axM.set_title('gridMet vs Daymet')
axM = mapplot.mapPoint(
    figM, gsM[0, 1], lat, lon, corrNM, s=16, cb=True, vRange=[0.5, 1])
axM.set_title('NLDAS vs Maurer')
axM = mapplot.mapPoint(
    figM, gsM[1, 1], lat, lon, corrMD, s=16, cb=True, vRange=[0.5, 1])
axM.set_title('Maurer vs Daymet')
axM = mapplot.mapPoint(
    figM, gsM[2, 1], lat, lon, corrND, s=16, cb=True, vRange=[0.5, 1])
axM.set_title('NLDAS vs Daymet')

figM.show()
