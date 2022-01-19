
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch

from hydroDL.model import waterNetTestC

import pandas as pd
import os

dataName = 'B5Y09-00955'
DF = dbBasin.DataFrameBasin(dataName)
codeLst = ['00955']
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

# model
nh = 16
nr = 5
model = waterNetTestC.Wn0110C2(nh, len(varXC), nr, nc=nc)
model = model.cuda()
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D().cuda()

sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
sizeLst = trainBasin.getSize(dataTup1)
[rho, nbatch] = batchSize
nIterEp = int(np.ceil((ns*nt)/(nbatch*rho)))
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
lossLst = list()

# random subset
model.train()
for ep in range(1, 1001):
    for iter in range(nIterEp):
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
        qP, cP = model(xT, xcT)
        lossQ = lossFun(qP, yT[nr-1:, :, 0])
        loss = lossQ
        lossCLst = list()
        for k in range(nc):
            lossC = lossFun(cP[:, :, k], yT[nr-1:, :, k+1])
            lossCLst.append(lossC)
            loss = loss+lossC
        optim.zero_grad()
        loss.backward()
        optim.step()
        strP = '{} {} {:.3f}'.format(ep, iter, lossQ.item())
        for lossC in lossCLst:
            strP = strP + ' {:.3f}'.format(lossC.item())
        print(strP)
    if (ep) % 50 == 0:
        modelFile = os.path.join(
            saveDir, 'wn0110C-00955-{}-ep{}'.format(dataName, ep))
        torch.save(model.state_dict(), modelFile)

lossFile = os.path.join(saveDir, 'loss-{}'.format(dataName))
pd.DataFrame(lossLst).to_csv(lossFile, index=False, header=False)
