
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
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(df.t, df.q[:, k, 0], '-k', label='ours')
ax.plot(dfN.t, dfN.q[:, k, 0], '-r', label='camels')
ax.legend()
ax.set_title(siteNoLst[k])
ax.set_xlim([t2dt(20120101), t2dt(20160101)])
# ax.set_ylim([0,20])
ax.set_yscale('log')
fig.show()

k = 100
siteNo = siteNoLst[k]
k1 = dfN.siteNoLst.index(siteNo)
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(df.t, df.q[:, k, 0], '-k', label='ours')
ax.plot(dfN.t, dfN.q[:, k1, 0], '-r', label='camels')
ax.legend()
ax.set_title(siteNoLst[k])
ax.set_xlim([t2dt(20000101), t2dt(20040101)])
# ax.set_ylim([0,20])
ax.set_yscale('log')
fig.show()

# map of difference
q = df.extractSubset(df.q, subsetName='benchmark')
qN = dfN.extractSubset(dfN.q, subsetName='benchmark')
rmse = utils.stat.calRmse(q, qN)
corr = utils.stat.calCorr(q, qN)
nse = utils.stat.calNash(q, qN)


lat, lon = df.getGeo()
figM = plt.figure()
gsM = gridspec.GridSpec(1, 2)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, rmse[:, 0], s=16, cb=True)
axM = mapplot.mapPoint(figM, gsM[0, 1], lat, lon, rmse[:, 1], s=16, cb=True)
figM.show()

np.where(corr < 0.99)

k = 100
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(q[:, k, 0], '-k', label='ours')
ax.plot(qN[:, k, 0], '-r', label='camels')
ax.twinx().plot(q[:, k, 0]-qN[:, k, 0], '--g', label='diff')
ax.legend()
ax.set_title(siteNoLst[k])
# ax.set_xlim([t2dt(20000101), t2dt(20040101)])
# ax.set_ylim([0,20])
ax.set_yscale('log')
fig.show()

q[:, k, 0]-qN[:, k, 0]


np.corrcoef(q[:, k, 0], qN[:, k, 0])
