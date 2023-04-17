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

dataName = 'singleDaily'
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)

dm = dbVeg.DataModelVeg(df, subsetName='all')


nh = 16
nt = len(df.t)
P = torch.arange(nt, dtype=torch.float32).reshape(-1, 1) / torch.pow(
    365 * 5, torch.arange(0, nh, 2, dtype=torch.float32) / nh
)
wP = torch.zeros((1, nt, nh))
wP[:, :, 0::2] = torch.sin(P)
wP[:, :, 1::2] = torch.cos(P)

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

k = indTrain[0]

bT = 100
bS = 5
bL = 3
bM = 15
iT = np.random.choice(nt - bT)
tempS=matS[iT : iT + bT, k, iS]
tempS=tempS[~np.isnan(tempS).all(axis=1)]
np.random.randint(0, tempS.shape[0],bS)


count = np.sum(~np.isnan(df.y), axis=0)

len(np.where(count > 20)[0])
