import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS, rnn, crit
from hydroDL.data import gageII, usgs, transform
from hydroDL.post import axplot, figplot
from sklearn.linear_model import LinearRegression
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from hydroDL.new.model import flowPath

siteNo = '07060710'
code = '00955'
freq = 'D'
sn = 1

# load data
varF = gridMET.varLst+ntn.varLst
varC = usgs.varC
varQ = usgs.varQ
varLst = varF+varC+varQ
df = waterQuality.readSiteTS(siteNo, varLst=varLst)

# plot data
fig, axes = plt.subplots(3, 1, figsize=(16, 6))
axplot.plotTS(axes[0], df.index, df['runoff'].values, styLst='-*', cLst='bgr')
axplot.plotTS(axes[1], df.index, df['pr'].values, styLst='-*', cLst='bgr')
axplot.plotTS(axes[2], df.index, df[code].values, styLst='*', cLst='bgr')
fig.show()

# training / testing
yrTrain = [2000, 2005]
yr = df.index.year.values
indTrain = np.where((yr >= yrTrain[0]) & (yr < yrTrain[1]))[0]

# data
# varX = varF
varX = ['pr']
varY = ['runoff']
nx = len(varX)
ny = len(varY)
X = df[varX].values
Y = df[varY].values
# mtdX = waterQuality.extractVarMtd(varX)
# mtdY = waterQuality.extractVarMtd(varY)
# x, statX = transform.transInAll(X, mtdX)
# y, statY = transform.transInAll(Y, mtdY)
# x[np.isnan(x)] = -1
x = np.log(X+1)
y = np.log(Y+1)
xx = x[indTrain, :]
yy = y[indTrain, :]

# model = rnn.LstmModel(nx=nx, ny=ny, hiddenSize=256).cuda()
try:
    model = flowPath(nx, 256, 10).cuda()
except:
    pass
model = flowPath(nx, 256, 3).cuda()
lossFun = crit.RmseLoss().cuda()
optim = torch.optim.Adadelta(model.parameters())

nt = len(indTrain)
nbatch = 100
rho = 1000
nd = 365
# train
nEp = 300
nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - nbatch/nt)))
xTemp = np.ndarray([rho, nbatch, nx])
yTemp = np.ndarray([rho, nbatch, ny])
lossLst = list()
for iEp in range(1, nEp + 1):
    t0 = time.time()
    for k in range(nbatch):
        iR = np.random.randint(0, nt-rho)
        xTemp[:, k, :] = xx[iR:iR+rho, :]
        yTemp[:, k, :] = yy[iR:iR+rho, :]
    xT = torch.from_numpy(xTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    if iEp == 1:
        try:
            yP = model(xT, nd)
        except:
            pass
    optim.zero_grad()
    yP = model(xT, nd)
    loss = lossFun(yP, yT[nd-1:, :, :])
    loss.backward()
    optim.step()
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, loss.item(), ct)
    lossLst.append(loss.item())
    print(logStr)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(range(nEp), lossLst)
fig.show()

# test
xA = np.expand_dims(x, axis=1)
xF = torch.from_numpy(xA).float().cuda()
yF = model(xF, nd)
yO = yF.detach().cpu().numpy()[:, 0, :]
# yOut = transform.transOutAll(yO, mtdY, statY)

# plot
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
ax.plot(df.index[nd-1:], yO[:, 0], color='r')
# ax.plot(df.index, Y, color='k')
# ax.plot( yy, color='k')
fig.show()
