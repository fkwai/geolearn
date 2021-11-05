
from sklearn.decomposition import PCA
from hydroDL.model import trainBasin
from hydroDL.data import dbBasin, gageII, gridMET
from hydroDL.master import basinFull
import numpy as np
from hydroDL import utils
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from hydroDL.model import waterNetGlobal
import importlib

dataName = 'HBN_Q90ref'
# dataName = 'temp'
DF = dbBasin.DataFrameBasin(dataName)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'LAI']
mtdX = ['skip' for k in range(4)]+['norm']
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
# mtdXC = dbBasin.io.extractVarMtd(varXC)
# mtdXC = ['QT' for var in varXC]
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()
DM3 = dbBasin.DataModelBasin(
    DF, subset='all', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM3.borrowStat(DM1)
dataTup3 = DM3.getData()

xc = dataTup1[1]
for k in range(17):
    fig, ax = plt.subplots(1, 1)
    ax.hist(xc[:, k], bins=30)
    fig.show()

pca = PCA(n_components=5)
xr = pca.fit(xc).transform(xc)
print(pca.explained_variance_ratio_)
fig, ax = plt.subplots(1, 1)
ax.plot(xr[:, 0], xr[:, 1], '*')
fig.show()
