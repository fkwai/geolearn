import importlib
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import matplotlib.gridspec as gridspec
from hydroDL.post import axplot, figplot, mapplot

dataName = 'B5Y09a'
DF = dbBasin.DataFrameBasin(dataName)

matMean = np.nanmean(DF.c, axis=0)
lat, lon = DF.getGeo()

figM = plt.figure()
gsM = gridspec.GridSpec(2, 2)
for k, var in enumerate(DF.varC):
    iy, ix = utils.index2d(k, 4, 2)
    axM = mapplot.mapPoint(
        figM, gsM[iy, ix], lat, lon,
        matMean[:, k], s=16, cb=True)
    axM.set_title('{}'.format(var))
figM.show()
