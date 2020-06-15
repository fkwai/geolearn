import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, wqLinear, wqRela
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from scipy import stats
import scipy

siteNo = '401733105392404'
# siteNo = '01364959'
codeLst = ['00915', '00955']

varX = gridMET.varLst
varY = ['00060']
dfX = waterQuality.readSiteX(siteNo, varX)
dfY = waterQuality.readSiteY(siteNo, varY)
dfC = waterQuality.readSiteY(siteNo, codeLst)
x = dfX['pr'].values
xA = dfX.values
y = dfY['00060'].values
nt = len(x)
rho = 365
matX = np.ones([nt-rho, rho+7])
for k in range(rho):
    matX[:, k] = x[k:nt-rho+k]
for k in range(5):
    matX[:, rho+k] = xA[rho:, k+2]

matY = y[rho:]

indV = np.where(~np.isnan(matY))[0]
xx = matX[indV, :]
yy = matY[indV]
g, res = scipy.optimize.nnls(xx, yy)
p = np.matmul(matX, g)

fig, ax = plt.subplots(1, 1)
ax.plot(p, '-r')
ax.plot(y[rho:], '-b')
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(g, '-g')
fig.show()
