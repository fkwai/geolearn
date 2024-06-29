import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
from hydroDL import utils
from hydroDL.post import mapplot, axplot, figplot
import matplotlib.pyplot as plt
from hydroDL.model import rnn, crit, trainBasin
import math
import torch
from torch import nn
from hydroDL.data import DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range
import torch.optim as optim
from hydroDL import kPath
import torch.optim.lr_scheduler as lr_scheduler
import dill

rho = 45
dataName = "singleDaily"
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)
df = dbVeg.DataFrameVeg(dataName)
dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
siteIdLst = df.siteIdLst
dm.trans(mtdDefault="minmax")
dataTup = dm.getData()
dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True)
x, xc, y, yc = dataEnd

varLst = ['VV', 'VH']
varLst = ['SR_B2', 'ndvi']
varLst = ['SR_B2', 'VV']

for var in varLst:
    k = df.varX.index(var)
    matB = ~np.isnan(x[:, :, k])
    count = matB.sum(axis=0)

    # plot hist
    fig, ax = plt.subplots(1, 1)
    ax.hist(count, bins=100)
    ax.set_title(var)
    fig.show()

# site based
varLst = ['SR_B2', 'VV']
# varLst=['SR_B2','ndvi']
for var in varLst:
    nt = len(df.t)
    varLst = ['VV', 'VH']
    xx = df.x
    k = df.varX.index(var)
    matB = ~np.isnan(xx[:, :, k])
    count = matB.sum(axis=0)
    temp=nt / count
    # rm inf
    temp[temp == np.inf] = 0
    # plot hist
    fig, ax = plt.subplots(1, 1)
    ax.hist(temp, bins=100)
    ax.set_title(var)
    fig.show()
