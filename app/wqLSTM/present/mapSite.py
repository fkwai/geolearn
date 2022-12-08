
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
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)

matB = (~np.isnan(DF.c)).astype(int).astype(float)
matS = np.nansum(matB, axis=-1) > 0
count = np.nansum(matS, axis=0)
lat, lon = DF.getGeo()

figM = plt.figure(figsize=(8, 5))
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(
    figM, gsM[0, 0], lat, lon, count, vRange=[200, 1000])
figM.show()
