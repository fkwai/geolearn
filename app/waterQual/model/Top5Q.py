from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS, rnn, crit
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# only look at 5 most complete HBN sites
# dataName = 'HBN'
# wqData = waterQuality.DataModelWQ(dataName)
# siteNoHBN = wqData.info['siteNo'].unique()
# indRm = wqData.indByComb(['00010', '00095'])
# info = wqData.info
# info = info.drop(indRm)
# tabCount = info.groupby('siteNo').count()
# siteNoLst = tabCount.nlargest(5, 'date').index.tolist()
# waterQuality.DataModelWQ.new('HBN5', siteNoLst)

dataName = 'HBN5'
# dataName = 'temp10'
wqData = waterQuality.DataModelWQ(dataName)

area = wqData.g[:, 0]
areaMat = np.tile(area, [365, 1])
areaMat = np.expand_dims(areaMat, axis=2)
# wqData.q = wqData.q/areaMat

varX = gridMET.varLst
varXC = gageII.lstWaterQuality
varY = usgs.varQ
varYC = None

varTup = (varX, varXC, varY, varYC)

# dataTup = wqData.extractData(varTup=varTup)
# xR, xcR, yR, ycR = dataTup
# mtdX = ['log-norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm']
# x, statX = transform.transInAll(xR, mtdX)


dataTup, statTup = wqData.transIn(varTup=varTup)
(x, xc, y, yc) = dataTup
dataTup = trainTS.dealNaN(dataTup, [1, 1, 0, 0])
(statX,statXC,statY,statYC)=statTup

# concatenate all data
[nx, nxc, ny, nyc, nt, ns] = trainTS.getSize(dataTup)
xx = np.zeros([ns, nt, nx+nxc])
for k in range(ns):
    xTemp = dataTup[0][:, k, :]
    xcTemp = dataTup[1][k, :]
    temp = np.concatenate([xTemp, np.tile(xcTemp, [365, 1])], axis=-1)
    xx[k, :, :] = temp
xT = torch.from_numpy(xx).float().cuda()
yy = np.swapaxes(dataTup[2], 0, 1)
yT = torch.from_numpy(yy).float().cuda()

# xT = xT[0:1, :, :]
# yT = yT[0:1, :, :]

# train model
model = rnn.CudnnLstmModel(
    nx=nx+nxc, ny=ny+nyc, hiddenSize=128)
lossFun = crit.RmseLoss()
lossFun = lossFun.cuda()
model = model.cuda()
optim = torch.optim.Adadelta(model.parameters())
# optim = torch.optim.SGD(model.parameters(), lr=0.01)
lossLst = list()


# backup - subset only spatial
nbatch = 500
iterEp=int(np.ceil(np.log(0.01) / np.log(1 - nbatch / ns)))

for i in range(100):
    nbatch = 500
    iS = np.random.randint(0, ns, nbatch)
    xT = torch.from_numpy(xx[iS, :, :]).float().cuda()
    yT = torch.from_numpy(yy[iS, :, :]).float().cuda()
    model.zero_grad()
    optim.zero_grad()
    yP = model(xT)
    loss = lossFun(yP, yT)
    loss.backward()
    optim.step()
    pred = yP.detach().cpu().numpy()
    lossLst.append(loss.item())
    print(i, loss.item())


fig, ax = plt.subplots(1, 1)
# ax.plot(errLst, lossLst, '*')
ax.plot(range(len(lossLst)), lossLst)
fig.show()


xT = torch.from_numpy(xx).float().cuda()
yP = model(xT)
pred = yP.detach().cpu().numpy()
obs = yy
qP = transform.transOutAll(pred, ['log-norm'],  statLst=statY)
qT = transform.transOutAll(obs, ['log-norm'],  statLst=statY)
prcp = xx[:, :, 0]


fig, ax = plt.subplots(1, 1)
k = np.random.randint(0, ns)
# k=0
t2 = np.datetime64(wqData.info.iloc[k]['date'], 'D')
t1 = t2-np.timedelta64(365, 'D')
t = np.arange(t1, t2)
axplot.plotTS(ax, t, [pred[k, :], obs[k, :]])
# axplot.plotTS(ax.twinx(), t, [prcp[k,:]],cLst='g',styLst='--')
# axplot.plotTS(ax, t, [qP[k, :], qT[k, :]])
ax.set_title(k)
fig.show()


# backup - subset both spatial and temporal
# for i in range(100):
#     nbatch = 500
#     rho = 300
#     iS = np.random.randint(0, ns, nbatch)
#     iT = np.random.randint(0, 365-rho, nbatch)
#     xTemp = np.full([nbatch, rho, nx+nxc], 0)
#     yTemp = np.full([nbatch, rho, ny+nyc], 0)
#     for k in range(nbatch):
#         xTemp[k, :] = xx[k, iT[k]:iT[k]+rho, :]
#         yTemp[k, :] = yy[k, iT[k]:iT[k]+rho, :]
#     xT = torch.from_numpy(xTemp).float().cuda()
#     yT = torch.from_numpy(yTemp).float().cuda()
#     model.zero_grad()
#     optim.zero_grad()
#     yP = model(xT)
#     loss = lossFun(yP, yT)
#     loss.backward()
#     optim.step()
#     pred = yP.detach().cpu().numpy()
#     lossLst.append(loss.item())
#     print(i, loss.item())


# backup train for all
# for i in range(100):
#     xT = torch.from_numpy(xx).float().cuda()
#     yT = torch.from_numpy(yy).float().cuda()
#     model.zero_grad()
#     optim.zero_grad()
#     yP = model(xT)
#     loss = lossFun(yP, yT)
#     loss.backward()
#     optim.step()
#     pred = yP.detach().cpu().numpy()
#     lossLst.append(loss.item())
#     print(i, loss.item())
