
import copy
import collections
from gc import collect
from hydroDL.model.waterNet import convTS, sepPar
from hydroDL.model import trainBasin, crit
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

from hydroDL.model import waterNetTestC
import importlib

importlib.reload(waterNetTestC)
importlib.reload(crit)

dataName = 'B5Y09a'
DF = dbBasin.DataFrameBasin(dataName)
codeLst = ['00915']
nc = len(codeLst)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
mtdY = ['skip' for k in range(nc+1)]
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

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


# extract subset
# siteNo = '04063700'
siteNo = '06317000'
siteNoLst = DF.getSite(trainSet)
indS = siteNoLst.index(siteNo)
dataLst1 = list()
dataLst2 = list()
for dataLst, dataTup in zip([dataLst1, dataLst2],
                            [dataTup1, dataTup2]):
    for data in dataTup:
        if data is not None:
            if data.ndim == 3:
                data = data[:, indS:indS+1, :]
            else:
                data = data[indS:indS+1, :]
        dataLst.append(data)
dataTup1 = tuple(dataLst1)
dataTup2 = tuple(dataLst2)


# model
nh = 16
nr = 5
model = waterNetTestC.Wn0110C2(nh, len(varXC), nr, nc=nc)
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters())
# optim = torch.optim.Rprop(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()

[x, xc, y, yc] = dataTup
xcP = torch.from_numpy(xc).float().cuda()

# random subset
model.train()
for kk in range(100):
    batchSize = [1000, 100]
    sizeLst = trainBasin.getSize(dataTup1)
    [x, xc, y, yc] = dataTup1
    [rho, nbatch] = batchSize
    [nx, nxc, ny, nyc, nt, ns] = sizeLst
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
    mDict = copy.deepcopy(dict(model.state_dict()))

    qP, cP = model(xT, xcT)
    lossQ = lossFun(qP, yT[nr-1:, :, 0])
    loss = lossQ
    lossCLst = list()
    for k in range(nc):
        lossC = lossFun(cP[:, :, k], yT[nr-1:, :, k+1])
        lossCLst.append(lossC)
        loss = loss+lossC
    # with torch.autograd.detect_anomaly():
    optim.zero_grad()
    loss.backward()
    optim.step()
    # mDict['fcR.0.weight'].sum()
    # model.state_dict()['fcR.0.weight'].sum()
    strP = '{} {:.3f}'.format(kk, lossQ.item())
    for lossC in lossCLst:
        strP = strP + ' {:.3f}'.format(lossC.item())
    print(strP)
    b = False
    for name, p in model.named_parameters():
        if p.isnan().any():
            print(kk, name)
            b = True
    if b:
        print(kk, 'break2')
        model.load_state_dict(mDict)
        break


t = DF.getT(testSet)
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
yT = torch.from_numpy(y).float().cuda()
qOut, cOut = model(xP, xcP)
qP = qOut.detach().cpu().numpy()
cP = cOut.detach().cpu().numpy()
lossQ = lossFun(qOut, yT[nr-1:, :, 0])
lossC = lossFun(cOut, yT[nr-1:, :, 1])
print(lossQ.item(), lossC.item())


# load LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=False, ep=1000)
yL = yL[:, indS, :]
yO = y[:, :, 0]
sd = 500
utils.stat.calNash(yL[sd:, :], yO[sd:, :])
utils.stat.calRmse(yL[sd:, :], yO[sd:, :])
utils.stat.calNash(qP[sd:, :], yO[sd+nr-1:, :])
utils.stat.calRmse(qP[sd:, :], yO[sd+nr-1:, :])

utils.stat.calNash(cP[sd:, :, 0], y[sd+nr-1:, :, 1])
utils.stat.calCorr(cP[sd:, :, 0], y[sd+nr-1:, :, 1])
utils.stat.calRmse(cP[sd:, :, 0], y[sd+nr-1:, :, 1])

k = 0
fig, axes = plt.subplots(1+nc, 1, sharex=True)
axes[0].plot(t[nr-1:], qP[:, k], '-r')
axes[0].plot(t, y[:, k, 0], '-k')
for k in range(nc):
    print(k)
    axes[k+1].plot(t[nr-1:], cP[:, 0, k], '-r')
    axes[k+1].plot(t, y[:, 0, k+1], '*k')
fig.show()
