import matplotlib.gridspec as gridspec
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from hydroDL import utils
import os
from hydroDL.model import trainBasin, crit, waterNetTest
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
import pandas as pd
from hydroDL.model import waterNetTest, waterNet
from hydroDL.master import basinFull
import importlib

importlib.reload(waterNetTest)
importlib.reload(crit)

dataName = 'QN90ref'
DF = dbBasin.DataFrameBasin(dataName)

DF.varG
varLst = ['BAS_COMPACTNESS', 'PRECIP_SEAS_IND', 'SLOPE_PCT',
          'ROCKDEPAVE', 'PLANTNLCD06','DEVNLCD06']
lat, lon = DF.getGeo()


figM = plt.figure(figsize=(12, 8))
gsM = gridspec.GridSpec(3, 2)
for k, var in enumerate(varLst):
    indG = DF.varG.index(var)
    j, i = utils.index2d(k, 3, 2)
    axM = mapplot.mapPoint(figM, gsM[j, i], lat, lon, DF.g[:, indG])
    axM.set_title(var)
figM.show()
