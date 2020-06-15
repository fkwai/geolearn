import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, wqLinear, wqRela
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model import rnn, crit
import time

siteNo = '401733105392404'
codeLst = ['00915', '00940', '00955']

varX = gridMET.varLst
varY = ['00060']
dfX = waterQuality.readSiteX(siteNo, varX)
dfY = waterQuality.readSiteY(siteNo, varY)

matX1 = dfX[dfX.index < np.datetime64('2000-01-01')].values
matY1 = dfY[dfY.index < np.datetime64('2000-01-01')].values
matX2 = dfX[dfX.index >= np.datetime64('2000-01-01')].values
matY2 = dfY[dfY.index >= np.datetime64('2000-01-01')].values

nx = len(varX)
ny = len(varY)
ind1 = np.where(~np.isnan(matY1))[0]
ind1 = ind1[ind1 > 365]
ind2 = np.where(~np.isnan(matY2))[0]
rho = 365
rhoF = 365
nh = 256
ns = 10


i2h = nn.Linear(nx + nh, nh)
h2h = nn.Linear(nh, rho)


x = np.ndarray([rho+rhoF, ns, nx])
y = np.ndarray([rhoF, ns, ny])
for k in range(ns):
    ind = ind1[np.random.randint(len(ind1))]
    x[:, k, :] = matX1[ind-rho-rhoF:ind, :]
    y[:, k, :] = matY1[ind-rhoF:ind, :]
xx = torch.from_numpy(x).float()
yy = torch.from_numpy(y).float()

nt, ns, nx = xx.shape
rhoF = nt-rho
h1 = torch.zeros(ns, nh)
zO = torch.zeros(rhoF, ns, 1)
sO = torch.zeros(rhoF, ns, rho)

for k in range(rho):
    t0=time.time()
    h1 = i2h(torch.cat((xx[k, :], h1), 1))
    h1 = F.relu(h1)
for k in range(rhoF):
    t0=time.time()
    h1 = i2h(torch.cat((xx[k, :], h1), 1))
    h1 = F.relu(h1)
    h2 = h2h(h1)
    h2 = F.softmax(h2)
    z = (xx[k:k+rho, :, 0].transpose(0, 1)*h2).sum(dim=1)
    zO[k, :, 0] = z
    sO[k, :, :] = h2


# fig, ax = plt.subplots(1, 1)
# ax.plot(dfY)
# fig.show()
