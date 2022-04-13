
import scipy
from sklearn.linear_model import LinearRegression
from hydroDL import utils
from sklearn.decomposition import PCA
import sklearn
import torch.nn.functional as F
import torch.nn as nn
import random
import os
from hydroDL.model import trainBasin, crit, waterNetTestC, waterNetTest
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import torch
import pandas as pd
import importlib
from hydroDL.utils import torchUtils
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from hydroDL.model.waterNet import WaterNet0119, sepPar, convTS

importlib.reload(waterNetTest)
# extract data
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']
# siteNoLst = ['04193500']
siteNo = '09163500'
dataName = 'siteNo'
# DF = dbBasin.DataFrameBasin.new(
#     dataName, [siteNo], varC=codeLst, varG=gageII.varLstEx)
# DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
# DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
DF = dbBasin.DataFrameBasin(dataName)

varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
mtdXC = ['skip' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()
DM0 = dbBasin.DataModelBasin(
    DF, subset='all', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM0.borrowStat(DM1)
dataTup0 = DM0.getData()

# check data plot
# number of complete data
DM = DM0
matNan = np.isnan(DM.y[:, 0, 1:])
ind = np.where(np.sum(matNan, axis=-1) == 0)[0]
temp = np.full([DM.y.shape[0], DM.y.shape[-1]], np.nan)
temp[ind, :] = DM.y[ind, 0, :]
labelLst = ['Q and P'] +\
    [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
fig, axes = figplot.multiTS(DM.t, [DM.y[:, 0, :], temp], labelLst=labelLst)
ax = axes[0].twinx()
ax.plot(DM.t, DM.x[:, 0, 0], 'b')
ax.invert_yaxis()
fig.show()


importlib.reload(waterNetTest)
importlib.reload(crit)

sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
nh = 16
nr = 5
nc = len(codeLst)
model = waterNetTest.Wn0119solo(nh, nr, nc)
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D().cuda()

# train
for ep in range(1, 100):
    [rho, nbatch] = batchSize
    iS = np.random.randint(0, ns, [nbatch])
    iT = np.random.randint(0, nt-rho, [nbatch])
    xTemp = np.full([rho, nbatch, nx], np.nan)
    yTemp = np.full([rho, nbatch, ny], np.nan)
    if x is not None:
        for k in range(nbatch):
            xTemp[:, k, :] = x[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if y is not None:
        for k in range(nbatch):
            yTemp[:, k, :] = y[iT[k]+1:iT[k]+rho+1, iS[k], :]
    xT = torch.from_numpy(xTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    model.zero_grad()
    yP = model(xT)
    loss = lossFun(yP, yT[nr-1:, :, 0])
    optim.zero_grad()
    loss.backward()
    optim.step()
    # torchUtils.ifNan(model)
    print(ep, loss.item())

# test
model.eval()
labelLst = ['Q and P'] +\
    [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
for dataTup, t in zip([dataTup1, dataTup2], [DM1.t, DM2.t]):
    [x, xc, y, yc] = dataTup
    xP = torch.from_numpy(x).float().cuda()
    nt, ns, _ = y.shape
    yOut = model(xP)
    yP = yOut.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1)
    axplot.plotTS(ax, t[nr-1:], [y[nr-1:, 0, 0], yP[:, 0]])
    fig.show()

# decomposition
varY = codeLst
mtdY = ['scale' for code in codeLst]
CM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=None, varXC=None, varY=varY, varYC=None)
CM1.trans(mtdY=mtdY)
[_, _, c1, _] = CM1.getData()
CM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=None, varXC=None, varY=varY, varYC=None)
CM2.borrowStat(CM1)
[_, _, c2, _] = CM2.getData()
CM0 = dbBasin.DataModelBasin(
    DF, subset='all', varX=None, varXC=None, varY=varY, varYC=None)
CM0.borrowStat(CM1)
[_, _, c0, _] = CM0.getData()

dataTup = dataTup1
[x, xc, y, yc] = dataTup
xP = torch.from_numpy(x).float().cuda()
yOut, qOut = model(xP, outQ=True)
QpO, QsO, QgO = qOut
Qa = yOut.detach().cpu().numpy()[:, 0]
Qp = QpO.detach().cpu().numpy()[:, 0, :]
Qs = QsO.detach().cpu().numpy()[:, 0, :]
Qg = QgO.detach().cpu().numpy()[:, 0, :]

# reg training
cMat = np.zeros([nh*3, nc])
q1 = np.concatenate([Qp, Qs, Qg], axis=-1)/Qa[:, None]
# qAll = np.concatenate([Qp, Qs, Qg], axis=-1)/y[nr-1:, 0,:]
for k in range(nc):
    data = c1[nr-1:, 0, k]
    [c, q] = utils.rmNan([data, q1], returnInd=False)
    a, r = scipy.optimize.nnls(q, c)
    cMat[:, k] = a
    out = np.sum(q1*a, axis=1)
    utils.stat.calCorr(out, c1[nr-1:, 0, k])
cp1 = np.matmul(q1, cMat)

# training plot
labelLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
fig, axes = figplot.multiTS(
    DM1.t[nr-1:], [cp1, c1[nr-1:, 0, :]], labelLst=labelLst)
fig.show()

# testing
dataTup = dataTup2
[x, xc, y, yc] = dataTup
xP = torch.from_numpy(x).float().cuda()
yOut, qOut = model(xP, outQ=True)
QpO, QsO, QgO = qOut
Qa = yOut.detach().cpu().numpy()[:, 0]
Qp = QpO.detach().cpu().numpy()[:, 0, :]
Qs = QsO.detach().cpu().numpy()[:, 0, :]
Qg = QgO.detach().cpu().numpy()[:, 0, :]
q2 = np.concatenate([Qp, Qs, Qg], axis=-1)/Qa[:, None]
cp2 = np.matmul(q2, cMat)
# testing plot
labelLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
fig, axes = figplot.multiTS(
    DM2.t[nr-1:], [cp2, c2[nr-1:, 0, :]], labelLst=labelLst)
fig.show()
utils.stat.calCorr(cp2, c2[nr-1:, 0, :])

cMat[:, 4]

# pca and end members
pca = PCA(5)
matNan = np.isnan(c0)
ind = np.where(np.sum(matNan, axis=-1) == 0)[0]
data = c0[ind, 0, :]
pca.fit(data)
print(pca.explained_variance_ratio_)
out = pca.transform(data)
cpout = pca.transform(cMat[:nh, :])
csout = pca.transform(cMat[nh:nh*2, :])
cgout = pca.transform(cMat[nh*2:nh*3, :])
fig, ax = plt.subplots(1, 1)
ax.plot(out[:, 0], out[:, 1], 'k.')
ax.plot(cpout[:, 0], cpout[:, 1], 'g*', label='surface EM')
ax.plot(csout[:, 0], csout[:, 1], 'r*', label='shallow EM')
ax.plot(cgout[:, 0], cgout[:, 1], 'b*', label='deep EM')
ax.legend()
fig.show()

# heat map of each EM
fig, axes = plt.subplots(nc, 1)
for k, code in enumerate(codeLst):
    temp = cMat[:, k]
    axplot.plotHeatMap(axes[k], temp.reshape(nh, 3).T)
    _ = axes[k].set_xticklabels([])
    _ = axes[k].set_yticklabels([])
    axes[k].set_title(usgs.codePdf.loc[code]['shortName'])
fig.show()
importlib.reload(axplot)

np.sum(Qp, axis=0)/np.sum(Qa, axis=0)
temp = np.concatenate(
    [np.mean(Qp/Qa[:, None], axis=0),
     np.mean(Qs/Qa[:, None], axis=0),
     np.mean(Qg/Qa[:, None], axis=0)])
fig, ax = plt.subplots(1, 1)
axplot.plotHeatMap(ax, temp.reshape(nh, 3).T*100, fmt='{:.1f}')
fig.show()
