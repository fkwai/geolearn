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

# training
# dataName = 'temp10'
# wqData = waterQuality.DataModelWQ(dataName)
# masterName = basins.wrapMaster(
#     dataName=dataName, trainName='first80', batchSize=[
#         None, 100], outName='temp10-Q', varYC=None, nEpoch=100)
# # basins.trainModelTS(masterName)

dataName = 'temp10'
outName = 'temp10-Q'
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
    nx=nx+nxc, ny=ny+nyc, hiddenSize=64)
lossFun = crit.RmseLoss()
lossFun = lossFun.cuda()
model = model.cuda()
optim = torch.optim.Adadelta(model.parameters())
# optim = torch.optim.SGD(model.parameters(), lr=0.01)
lossLst = list()

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

for i in range(1000):
    xT = torch.from_numpy(xx).float().cuda()
    yT = torch.from_numpy(yy).float().cuda()
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
qP= transform.transOutAll(pred, ['log-norm'],  statLst=statY)
qT= transform.transOutAll(obs, ['log-norm'],  statLst=statY)
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



# testing using training data
# dataTup, statTup = wqData.transIn(
#     subset=dictP['trainName'], varTup=(varX, varXC, varY, varYC))
# dataTup = trainTS.dealNaN(dataTup, dictP['optNaN'])
# info = wqData.subsetInfo(dictP['trainName'])


# # calculate error
# matT = np.full([ns, 365], np.nan)
# matObs = np.full([ns, 365], np.nan)
# matPred = np.full([ns, 365], np.nan)
# for k in range(ns):
#     x = dataTup[0][:, k, :]
#     xc = dataTup[1][k, :]
#     xx = np.concatenate([x, np.tile(xc, [365, 1])], axis=-1)
#     xx = np.expand_dims(xx, axis=0)
#     xT = torch.from_numpy(xx).float().cuda()
#     if torch.cuda.is_available():
#         xT = xT.cuda()
#     yT = model(xT)
#     pred = yT.detach().cpu().numpy().flatten()
#     obs = dataTup[2][:, k, :].flatten()
#     t2 = np.datetime64(info.iloc[k]['date'], 'D')
#     t1 = t2-np.timedelta64(365, 'D')
#     t = np.arange(t1, t2)
#     matT[k, :] = t
#     matObs[k, :] = obs
#     matPred[k, :] = pred


# err = np.nanmean(np.nanmean((matPred - matObs)**2, axis=1))

# k = 500
# fig, ax = plt.subplots(1, 1)
# axplot.plotTS(ax, t, [matPred[k, :], matObs[k, :]])
# fig.show()
