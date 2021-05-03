import importlib
from hydroDL.master import basins
from hydroDL import kPath, utils
from hydroDL.model import trainTS, rnn, crit
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin

siteNo = '06800500'
codeLst = ['00915', '00925']

df = GLASS.readBasin(siteNo)

# plot data
df = dbBasin.readSiteTS(siteNo, varLst=codeLst, freq='D')
nc = len(codeLst)
fig, axes = plt.subplots(nc, 1, figsize=(12, 3))
for k, code in enumerate(codeLst):
    axplot.plotTS(axes[k], df.index, df[code].values, cLst='k')
fig.show()

# load data
varLst = ntn.varLst
# df = dbBasin.readSiteTS(siteNo, varLst=varLst, freq='D')
dfP = ntn.readBasin(siteNo, varLst=varLst, freq='D')
varP = varLst.copy()
varP.remove('distNTN')
fig, axes = plt.subplots(len(varP), 1, figsize=(12, 3))
axplot.plotTS(axes[0], df.index, df['Ca'].values, cLst='k')
axplot.plotTS(axes[1], df.index, df['distNTN'].values, cLst='k')
fig.show()

fig, axes = plt.subplots(len(varP), 1, figsize=(12, 3))
for k, var in enumerate(varP):
    axes[k].scatter(df.index, df[var], c=df['distNTN'], s=10, zorder=2)
    axes[k].plot(df.index, df[var], 'k-', zorder=1)
fig.show()
