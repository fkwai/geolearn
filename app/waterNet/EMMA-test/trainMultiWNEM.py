
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
import time
importlib.reload(waterNetTestC)

# extract data
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']

siteNoLst = ['09163500']
# siteNo = '04193500'
dataName = 'temp'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=codeLst, varG=gageII.varLstEx)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')


# def train(dataName, nm, codeLst):
DF = dbBasin.DataFrameBasin(dataName)
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
mtdY = ['skip'] + ['scale' for code in codeLst]
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC)
dataTup1 = DM1.getData()
sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
ng = xc.shape[-1]
nh = 16
nr = 5
nc = len(codeLst)
nm = nh
model = waterNetTestC.Wn0119EM(nh, ng, nr, nc, nm).cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.01)
# lossFun = crit.NashLoss3D().cuda()
lossFun = crit.LogLoss3D().cuda()


# train
torch.autograd.set_detect_anomaly(True)

for ep in range(1, 101):
    t0 = time.time()
    [rho, nbatch] = batchSize
    iS = np.random.randint(0, ns, [nbatch])
    iT = np.random.randint(0, nt-rho, [nbatch])
    xTemp = np.full([rho, nbatch, nx], np.nan)
    xcTemp = np.full([nbatch, nxc], np.nan)
    yTemp = np.full([rho, nbatch, ny], np.nan)
    ycTemp = np.full([nbatch, nyc], np.nan)
    if x is not None:
        for k in range(nbatch):
            xTemp[:, k, :] = x[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if y is not None:
        for k in range(nbatch):
            yTemp[:, k, :] = y[iT[k]+1:iT[k]+rho+1, iS[k], :]
    if xc is not None:
        xcTemp = xc[iS, :]
    if yc is not None:
        ycTemp = yc[iS, :]
    xT = torch.from_numpy(xTemp).float().cuda()
    xcT = torch.from_numpy(xcTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    ycT = torch.from_numpy(ycTemp).float().cuda()

    model.zero_grad()
    yP = model(xT, xcT)
    # loss = lossFun(yP[:, :, :], yT[nr-1:, :, :])
    lossQ = lossFun(yP[:, :, 0:1], yT[nr-1:, :, 0:1])
    lossC = lossFun(yP[:, :, 1:], yT[nr-1:, :, 1:])
    loss = lossQ*lossC
    optim.zero_grad()
    loss.backward()
    optim.step()
    # torchUtils.ifNan(model)
    print('{} {:.2f} {:.2f} {:.2f}'.format(
        ep, lossQ.item(), lossC.item(), time.time()-t0))


xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yOut = model(xP, xcP)
yP = yOut.detach().cpu().numpy()
for k, siteNo in enumerate(siteNoLst):
    fig, axes = figplot.multiTS(DM1.t[nr-1:], [yP[:, k, :], y[nr-1:, k, :]])
    utils.stat.calCorr(yP[:, k, :], y[nr-1:, k, :])
    fig.show()


# save model
# saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTempEM'
# modelFile = os.path.join(
#     saveDir, 'wn0119Multi-{}-ep{}-nm{}'.format(dataName, ep, nm))
# torch.save(model.state_dict(), modelFile)


# train('09163500', 16, codeLst)
# train('09163500', 8, codeLst)
# train('09163500', 4, codeLst)
# train('09163500', 1, codeLst)
# train('04193500', 16, codeLst)
# train('04193500', 1, codeLst)
