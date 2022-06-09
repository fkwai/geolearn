
import torch.nn.functional as F
import torch.nn as nn
import random
import os
from hydroDL.model import trainBasin, crit, waterNetTest
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
import time

# extract data
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']

siteNo = '09163500'
# dataName = 'temp'
# DF = dbBasin.DataFrameBasin.new(
#     dataName, siteNoLst, varC=codeLst, varG=gageII.varLstEx)
# DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
# DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
DF = dbBasin.DataFrameBasin(siteNo)

varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
mtdX = ['skip' for k in range(4)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
mtdY = ['skip'] + ['scale' for code in codeLst]
varXC = gageII.varLstEx
mtdXC = ['norm' for var in varXC]
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


# check data plot
labelLst = ['Q and P'] +\
    [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
fig, axes = figplot.multiTS(DM1.t, DM1.y[:, 0, :], labelLst=labelLst)
ax = axes[0].twinx()
ax.plot(DM1.t, DM1.x[:, 0, 0], 'b')
ax.invert_yaxis()
fig.show()

sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
nh = 16
nr = 5
nc = len(codeLst)
ng = xc.shape[-1]

model = WaterNet0119(nh, ng, nr).cuda()
lossFun = crit.LogLoss3D().cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.01)
nm = nh

fcC = nn.Sequential(
    nn.Linear(ng, 256),
    nn.Tanh(),
    nn.Dropout(),
    nn.Linear(256, nm*nc*3)).cuda()

for ep in range(1, 101):
    t0 = time.time()
    # wrap up data
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

    # append on a waterNet model
    Qout, (QpR, QsR, QgR), (SfT, SsT, SgT) = model(xT, xcT, outStep=True)
    c = fcC(xcT)

    cLst = ['skip', 'skip', 'skip']
    [cpT, csT, cgT] = sepPar(c, nm*nc, cLst)
    cp = cpT.view(-1, nm, nc)
    cs = csT.view(-1, nm, nc)
    cg = cgT.view(-1, nm, nc)
    cp = torch.relu(torch.exp(cp))
    cs = torch.relu(torch.exp(cs))
    cg = torch.relu(torch.exp(cg))
    ntTemp = Qout.shape[0]
    CpR = (QpR/Qout[:, :, None])[:, :, :, None] * cp.repeat(ntTemp, 1, 1, 1)
    CsR = (QsR/Qout[:, :, None])[:, :, :, None] * cs.repeat(ntTemp, 1, 1, 1)
    CgR = (QgR/Qout[:, :, None])[:, :, :, None] * cg.repeat(ntTemp, 1, 1, 1)
    Cout = torch.sum(CpR+CsR+CgR, dim=2)
    yOut = torch.cat([Qout[..., None], Cout], dim=-1)
    lossQ = lossFun(yOut[:, :, 0:1], yT[nr-1:, :, 0:1])
    lossC = lossFun(yOut[:, :, 1:], yT[nr-1:, :, 1:])
    loss = lossQ*lossC
    optim.zero_grad()
    loss.backward()
    optim.step()
    print('{} {:.2f} {:.2f} {:.2f}'.format(
        ep, lossQ.item(), lossC.item(), time.time()-t0))


fcC(xcT[1:2, :])-fcC(xcT[0:1, :])
ff = nn.Linear(ng, 256).cuda()
ff(xcT[1:2, :])-ff(xcT[0:1, :])
xcT[0:1, :]-xcT[1:2, :]

a = torch.rand(nm, nc)
a.mean()
