import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
import matplotlib.gridspec as gridspec
from hydroDL.post import mapplot, axplot, figplot
from hydroDL.data import DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range


import matplotlib.pyplot as plt

dataName = 'singleDaily'
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)

lat, lon = df.lat, df.lon

count = (~np.isnan(df.y)).sum(axis=0)

varS = ['VV', 'VH']


def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, count)

    figP, axP = plt.subplots(3, 1, sharex=True)
    axT1, axT2, axT3 = [ax.twinx() for ax in axP]
    axP = np.array([ax for ax in axP] + [axT1, axT2, axT3])

    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    axP1, axP2, axP3, axT1, axT2, axT3 = axP
    axP1.plot(df.t, df.y[:, iP], 'ro')
    axP2.plot(df.t, df.y[:, iP], 'ro')
    axP3.plot(df.t, df.y[:, iP], 'ro')
    axT1.plot(df.t, df.x[:, iP, df.varX.index('VH')], '*')
    axT2.plot(df.t, df.x[:, iP, df.varX.index('ndvi')], '*')
    axT3.plot(df.t, df.x[:, iP, df.varX.index('Lai')], '*')


figplot.clickMap(funcM, funcP)

import matplotlib

matplotlib.rcParams.update({'font.size': 11})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})
matplotlib.rcParams.update({'legend.fontsize': 11})

iP = 3
iP = 1
figP, axP = plt.subplots(4, 1, sharex=True)
axP1, axP2, axP3, axP4 = axP
axP1.plot(df.t, df.y[:, iP], 'ro', label='LFMC')
axP2.plot(df.t, df.x[:, iP, df.varX.index('VH')], '*', label='VH')
axP3.plot(df.t, df.x[:, iP, df.varX.index('ndvi')], 'g*', label='NDVI')
axP4.plot(df.t, df.x[:, iP, df.varX.index('Lai')], 'g*', label='LAI')
axP1.legend()
axP2.legend()
axP3.legend()
axP4.legend()
figP.show()
figP.subplots_adjust(wspace=0, hspace=0)

tV = df.t[~np.isnan(df.y[:, iP]).flatten()]
iP=3
tC = tV[23]
figP, axP = plt.subplots(4, 1, sharex=True)
axP1, axP2, axP3, axP4 = axP
axP1.plot(df.t, df.y[:, iP], 'ro', label='LFMC')

v1 = df.x[:, iP, df.varX.index('VH')]
axP2.plot(df.t, v1, '*', label='VH')
ylim = axP2.get_ylim()
t1= df.t[~np.isnan(v1)]
for tt in t1:
    axP2.plot([tt,tt], ylim, '-',color='grey',linewidth=0.5)

v2 = df.x[:, iP, df.varX.index('ndvi')]
axP3.plot(df.t, df.x[:, iP, df.varX.index('ndvi')], 'g*', label='NDVI')
ylim = axP3.get_ylim()
t2= df.t[~np.isnan(v2)]
for tt in t2:
    axP3.plot([tt,tt], ylim, '-',color='grey',linewidth=0.5)

axP4.plot(df.t, df.x[:, iP, df.varX.index('Lai')], 'g*', label='LAI')

axP1.legend()
axP2.legend()
axP3.legend()
axP4.legend()
axP1.set_xlim([tC - np.timedelta64(45, 'D'), tC + np.timedelta64(45, 'D')])
figP.subplots_adjust(wspace=0, hspace=0)

figP.show()
