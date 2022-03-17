
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS

import warnings

DFN = dbBasin.DataFrameBasin('G200N')
DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.newC

bins = 50

# raw data
fig, axes = plt.subplots(4, 5)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 4, 5)
    ic = DFN.varC.index(code)
    data = DFN.c[:, :, ic].flatten()
    ub = np.nanpercentile(data, 90)
    lb = np.nanpercentile(data, 10)
    b = (data < ub) & (data > lb)
    ax = axes[j, i]
    ax.hist(data[b], bins=bins, density=True)
    # ax.set_xlim([-5, 5])
    axplot.titleInner(ax, code)
fig.show()

# global norm
varY = codeLst
mtdY = dbBasin.io.extractVarMtd(varY)
y, statY = transform.transIn(DF.c, mtdLst=mtdY)
fig, axes = plt.subplots(4, 5)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 4, 5)
    ic = codeLst.index(code)
    data = y[:, :, ic].flatten()
    ax = axes[j, i]
    ax.hist(data, bins=bins, density=True)
    ax.set_xlim([-5, 5])
    axplot.titleInner(ax, code)
fig.show()

# local norm
fig, axes = plt.subplots(4, 5)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 4, 5)
    ic = DFN.varC.index(code+'-N')
    data = DFN.c[:, :, ic].flatten()
    ax = axes[j, i]
    ax.hist(data, bins=bins, density=True)
    ax.set_xlim([-5, 5])
    axplot.titleInner(ax, code)
fig.show()
