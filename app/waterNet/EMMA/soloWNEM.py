
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
from hydroDL import utils
importlib.reload(waterNetTestC)
# extract data
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']

# siteNo = '09163500'
siteNo = '04193500'
dataName = siteNo
DF = dbBasin.DataFrameBasin.new(
    dataName, [siteNo], varC=codeLst, varG=gageII.varLstEx)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
DF = dbBasin.DataFrameBasin(dataName)

varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
mtdY = ['skip'] + ['scale' for code in codeLst]
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
labelLst = ['Q and P'] +\
    [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
fig, axes = figplot.multiTS(DM1.t, DM1.y[:, 0, :], labelLst=labelLst)
ax = axes[0].twinx()
ax.plot(DM1.t, DM1.x[:, 0, 0], 'b')
ax.invert_yaxis()
fig.show()
# number of complete data
matNan = np.isnan(DM1.y[:, 0, 1:])
ind = np.where(np.sum(matNan, axis=-1) == 0)[0]
len(ind)

importlib.reload(waterNetTestC)
importlib.reload(crit)

sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
nh = 16
nr = 5
nc = len(codeLst)
nm = 4
model = waterNetTestC.Wn0119EMsolo(nh, nr, nc, nm)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
lossFun = crit.LogLoss3D().cuda()

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
    # loss = lossFun(yP[:, :, :], yT[nr-1:, :, :])
    lossQ = lossFun(yP[:, :, 0:1], yT[nr-1:, :, 0:1])
    lossC = lossFun(yP[:, :, :], yT[nr-1:, :, :])
    loss = lossC
    optim.zero_grad()
    loss.backward()
    optim.step()
    # torchUtils.ifNan(model)
    print(ep, lossQ.item(), lossC.item())

# test
model.eval()
labelLst = ['Q and P'] +\
    [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
for dataTup, t in zip([dataTup1, dataTup2], [DM1.t, DM2.t]):
    [x, xc, y, yc] = dataTup
    xP = torch.from_numpy(x).float().cuda()
    xcP = torch.from_numpy(xc).float().cuda()
    nt, ns, _ = y.shape
    yOut = model(xP)
    yP = yOut.detach().cpu().numpy()
    labelLst = ['Q'] +\
        [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
    dataPlot = np.stack([y[nr-1:, 0, :], yP[:, 0, :]], axis=-1)
    fig, axes = plt.subplots(nc+1, 1)
    utils.stat.calCorr(y[nr-1:, 0, :], yP[:, 0, :])
    fig, axes = figplot.multiTS(
        t[nr-1:], [y[nr-1:, 0, :], yP[:, 0, :]], labelLst=labelLst)
    fig.show()


cp = torch.relu(torch.exp(model.cp)-1).detach().cpu().numpy()
cs = torch.relu(torch.exp(model.cs)-1).detach().cpu().numpy()
cg = torch.relu(torch.exp(model.cg)-1).detach().cpu().numpy()


# pca and end members
pca = PCA(5)
[x, xc, y, yc] = dataTup0
temp = y[:, 0, 1:]
matNan = np.isnan(temp)
ind = np.where(np.sum(matNan, axis=-1) == 0)[0]
data = temp[ind, :]
pca.fit(data)
print(pca.explained_variance_ratio_)
out = pca.transform(data)
# cpout = pca.transform(cp)
# csout = pca.transform(cs)
# cgout = pca.transform(cg)
cpout = pca.transform(cp)
# cpout = pca.transform(np.zeros([1, nc]))
csout = pca.transform(cs)
cgout = pca.transform(cg)
fig, ax = plt.subplots(1, 1)
ax.plot(out[:, 0], out[:, 1], 'k.')
ax.plot(cpout[:, 0], cpout[:, 1], 'g*', label='surface EM')
ax.plot(csout[:, 0], csout[:, 1], 'r*', label='shallow EM')
ax.plot(cgout[:, 0], cgout[:, 1], 'b*', label='deep EM')
ax.legend()
fig.show()


# # heat map of each EM
# fig, axes = plt.subplots(nc, 1)
# for k, code in enumerate(codeLst):
#     temp = cMat[:, k]
#     axplot.plotHeatMap(axes[k], temp.reshape(nh/nm, 3).T)
#     _ = axes[k].set_xticklabels([])
#     _ = axes[k].set_yticklabels([])
#     axes[k].set_title(usgs.codePdf.loc[code]['shortName'])
# fig.show()
# importlib.reload(axplot)

# np.sum(Qp, axis=0)/np.sum(Qa, axis=0)
# temp = np.concatenate(
#     [np.mean(Qp/Qa[:, None], axis=0),
#      np.mean(Qs/Qa[:, None], axis=0),
#      np.mean(Qg/Qa[:, None], axis=0)])
# fig, ax = plt.subplots(1, 1)
# axplot.plotHeatMap(ax, temp.reshape(nh, 3).T*100, fmt='{:.1f}')
# fig.show()
