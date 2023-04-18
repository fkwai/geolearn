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


dataName = 'singleDaily'
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)

dm = dbVeg.DataModelVeg(df, subsetName='all')

varS = ['VV', 'VH', 'vh_vv']
varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
varM = ['Fpar', 'Lai']

iS = [df.varX.index(var) for var in varS]
matS = df.x[:, :, iS]
iL = [df.varX.index(var) for var in varL]
matL = df.x[:, :, iL]
iM = [df.varX.index(var) for var in varM]
matM = df.x[:, :, iM]

countS = np.sum(~np.isnan(matS), axis=0)[:, 0]
countL = np.sum(~np.isnan(matL), axis=0)[:, 0]
countM = np.sum(~np.isnan(matM), axis=0)[:, 0]
count = np.sum(~np.isnan(df.y), axis=0)[:, 0]
ind = np.where((countS > 80) & (countL > 80) & (countM > 100) & (count > 15))[0]
nsite = len(ind)
nTrain = math.floor(len(ind) * 0.8)
indTrain = ind[torch.randperm(nsite)[:nTrain]]
indTest = np.setdiff1d(ind, indTrain)

# add date index
dm = DataModel(
    X=df.x[:, ind, :],
    XC=df.xc[
        ind,
    ],
    Y=df.y[:, ind, :],
)
dm1 = DataModel(
    X=df.x[:, indTrain, :],
    XC=df.xc[
        indTrain,
    ],
    Y=df.y[:, indTrain, :],
)
dm2 = DataModel(
    X=df.x[:, indTest, :],
    XC=df.xc[
        indTest,
    ],
    Y=df.y[:, indTest, :],
)

dm.trans(mtdDefault='norm')
dm1.borrowStat(dm)
dm2.borrowStat(dm)

dataTup1 = dm1.getData()
dataTup2 = dm2.getData()
rho = 45
dataEnd1, (iT1, jT1) = dataTs2Range(dataTup1, rho, returnInd=True)
dataEnd2, (iT2, jT2) = dataTs2Range(dataTup2, rho, returnInd=True)
t1 = df.t[iT1]
t2 = df.t[iT2]

x1, xc1, y1, yc1 = dataEnd1
k = indTrain[0]
bS = 8
bL = 8
bM = 15

# calculate
pSLst, pLLst, pMLst = list(), list(), list()
ns = yc1.shape[0]
nMat = np.zeros([ns, 3])
for k in range(ns):
    tempS = x1[:, k, iS]
    pS = np.where(~np.isnan(tempS).all(axis=1))[0]
    tempL = x1[:, k, iL]
    pL = np.where(~np.isnan(tempL).all(axis=1))[0]
    tempM = x1[:, k, iM]
    pM = np.where(~np.isnan(tempM).all(axis=1))[0]
    pSLst.append(pS)
    pLLst.append(pL)
    pMLst.append(pM)
    nMat[k, :] = [len(pS), len(pL), len(pM)]


# remove all empty
indNan = np.where(nMat == 0)[0]
indKeep = np.setdiff1d(np.arange(ns), indNan)
len(indNan)

x1 = x1[:, indKeep, :]
xc1 = xc1[indKeep, :]
yc1 = yc1[indKeep, :]
nMat = nMat[indKeep, :]
pSLst = [pSLst[k] for k in indKeep]
pLLst = [pLLst[k] for k in indKeep]
pMLst = [pMLst[k] for k in indKeep]
ns = yc1.shape[0]

ns
