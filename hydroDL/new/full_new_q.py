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
import torch.nn.functional as F
from hydroDL.new import fun


siteNo = '401733105392404'
code = '00955'
freq = 'D'


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
sn = 1
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
# y = np.log(Y+sn)
# x[np.isnan(x)] = -1
x = X
y = Y
xx = x[indTrain, :]
yy = y[indTrain, :]

# conv
nt = len(indTrain)
nbatch = 100
rho = 1000
aLst = np.exp(np.arange(0, 2, 0.1))
m = 30
nq = len(aLst)
nd = 365
tt = np.arange(nd)+1
qc = np.ndarray([nd, nq])
for k in range(nq):
    the = m
    qc[:, k] = fun.fdc(tt.T, aLst[k], the=the)

qcT = torch.from_numpy(qc.T[None, :, :]).float().cuda()


model = rnn.LstmModel(nx=nx, ny=nq, hiddenSize=256).cuda()
lossFun = crit.RmseLoss().cuda()
optim = torch.optim.Adadelta(model.parameters())

# train
nEp = 2000
# nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - nbatch*rho/nt)))
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
            yP = model(xT)
        except:
            pass
    optim.zero_grad()
    yP = model(xT)
    # yP2 = F.conv1d(yP.exp().permute([1, 2, 0]), qcT).permute(2, 0, 1).log()
    yP2 = F.conv1d(yP.permute([1, 2, 0]), qcT).permute(2, 0, 1)
    loss = lossFun(yP2[:, :, :], yT[nd-1:, :, :])
    loss.backward()
    optim.step()
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, loss.item(), ct)
    lossLst.append(loss.item())
    print(logStr)

torch.save(model, 'temp')
model = torch.load('temp')

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(range(nEp), lossLst)
fig.show()

k = 10
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(yP2[:, k, 0].detach().cpu().numpy())
ax.plot(yT[nd-1:, k, 0].detach().cpu().numpy())
fig.show()


# test
xA = np.expand_dims(x, axis=1)
xF = torch.from_numpy(xA).float().cuda()
yF = model(xF)
# yF2 = F.conv1d(yF.exp().permute([1, 2, 0]), qcT).permute(2, 0, 1)
yF2 = F.conv1d(yF.permute([1, 2, 0]), qcT).permute(2, 0, 1)
yO = yF2.detach().cpu().numpy()[:, 0, :]
# yOut = transform.transOutAll(yO, mtdY, statY)

# plot
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
# axplot.plotTS(ax, df.index, [y, yO], styLst='--', cLst='kr')
ax.plot(df.index, Y, '-k')
# ax.plot(df.index[nd-1:], yOut, '-r')
ax.plot(df.index[nd-1:], yO, '-r')
fig.show()

# plot
yG = yF[:, 0, :].exp().detach().cpu().numpy()
cLst = plt.cm.jet(np.linspace(0, 1, nq))

# inflow
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
for k in range(nq):
    ax.plot(df.index, yG[:, k], color=cLst[k])
fig.show()
# ax.set_xlim(np.datetime64('2015-01-01'), np.datetime64('2018-01-01'))

# fdc
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
for k in range(nq):
    ax.plot(tt, qc[:, k], color=cLst[k])
fig.show()

# obs
fig, axes = plt.subplots(3, 1, figsize=(16, 6))
axplot.plotTS(axes[0], df.index, df['00060'].values, styLst='-*', cLst='bgr')
axplot.plotTS(axes[1], df.index, df['pr'].values, styLst='-*', cLst='bgr')
axplot.plotTS(axes[2], df.index, df[code].values, styLst='*', cLst='bgr')
fig.show()
# for ax in fig.axes:
#     ax.set_xlim(np.datetime64('2015-01-01'), np.datetime64('2018-01-01'))
# fig.canvas.draw()

# calculate water age
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
qMat = np.zeros([len(df)-nd+1, nq])
for k in range(nq):
    a = np.convolve(np.flip(qc[:, k]), yG[:, k], 'valid')
    # b = np.log(a[:, None])
    # c = transform.transOutAll(a[:, None], ['norm'], statY)
    qMat[:, k] = a

# ax.plot(df.index[364:], b, color=cLst[k])
b = np.sum(qMat, axis=1)-sn
# c = transform.transOutAll(b[:, None], mtdY, statY)
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
ax.plot(df.index[364:], b, color='r')
ax.plot(df.index, Y, color='k')
# ax.plot( yy, color='k')
fig.show()
