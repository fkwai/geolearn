import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, wqLinear
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hydroDL.model import rnn, crit, trainTS
import time

siteNo = '02175000'
codeLst = ['00915', '00945', '00955']
# codeLst = ['00915', '00955']
nh = 256
batchSize = [365, 50]
if not waterQuality.exist(siteNo):
    wqData = waterQuality.DataModelWQ.new(siteNo, [siteNo])
wqData = waterQuality.DataModelWQ(siteNo)
varX = wqData.varF
varXC = wqData.varG
varY = [wqData.varQ[0]]
varYC = codeLst
varTup = (varX, varXC, varY, varYC)
dataTup, statTup = wqData.transIn(varTup=varTup)
dataTup = trainTS.dealNaN(dataTup, [1, 1, 0, 0])
sizeLst = trainTS.getSize(dataTup)
[nx, nxc, ny, nyc, nt, ns] = sizeLst

tabG = gageII.readData(varLst=varXC, siteNoLst=[siteNo])
tabG = gageII.updateCode(tabG)
dfX = waterQuality.readSiteX(siteNo, varX, nFill=5)
dfY = waterQuality.readSiteY(siteNo, varY)
dfYC = waterQuality.readSiteY(siteNo, varYC)

importlib.reload(rnn)
importlib.reload(crit)

model = rnn.AgeLSTM2(
    nx=nx+nxc, ny=ny, nyc=nyc, rho=365, nh=nh)
optim = torch.optim.Adadelta(model.parameters())
lossFun = crit.RmseLoss2D()
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()

# train
model.train()
model.zero_grad()
for k in range(500):
    # k=0
    t0 = time.time()
    xT, yT = trainTS.subsetRandom(dataTup, batchSize, sizeLst)
    if k == 0:
        try:
            yP = model(xT)
        except:
            print('first iteration failed again for CUDNN_STATUS_EXECUTION_FAILED ')
    yP = model(xT)
    loss = lossFun(yP, yT[-1, :, :])
    loss.backward()
    optim.step()
    model.zero_grad()
    print('{} {:.3f} {:.3f}'.format(k, loss, time.time()-t0))
    torch.cuda.empty_cache()


# test
statX, statXC, statY, statYC = statTup
xA = np.expand_dims(dfX.values, axis=1)
xcA = np.expand_dims(
    tabG.loc[siteNo].values.astype(np.float), axis=0)
mtdX = wqData.extractVarMtd(varX)
x = transform.transInAll(xA, mtdX, statLst=statX)
mtdXC = wqData.extractVarMtd(varXC)
xc = transform.transInAll(xcA, mtdXC, statLst=statXC)

yA = np.expand_dims(dfY.values, axis=1)
ycA = np.expand_dims(dfYC.values, axis=1)
mtdY = wqData.extractVarMtd(varY)
y = transform.transInAll(yA, mtdY, statLst=statY)
mtdYC = wqData.extractVarMtd(varYC)
yc = transform.transInAll(ycA, mtdYC, statLst=statYC)

(x, xc) = trainTS.dealNaN((x, xc), [1, 1])
nt = x.shape[0]
xT = torch.from_numpy(np.concatenate(
    [x, np.tile(xc, [nt, 1, 1])], axis=-1)).float()
if torch.cuda.is_available():
    xT = xT.cuda()

model = model.train(mode=False)
# yP, gate = model(xT)
yP, b, gate = model(xT)
yO = yP.detach().cpu().numpy()
# gate = gate.detach().cpu().numpy()

predY = transform.transOut(yO[:, :, 0], mtdY[0], statY[0])
predYC = transform.transOutAll(yO[:, :, 1:], mtdYC, statYC)
obsY = dfY.values
obsYC = dfYC.values

t = dfY.index.values
fig, axes = plt.subplots(4, 1)
axplot.plotTS(axes[0], t, [predY, obsY],
              styLst='---', cLst='rb')
axes[0].set_title('streamflow')
axes[0].set_xticks([])
codePdf = usgs.codePdf
for k, code in enumerate(codeLst):
    axplot.plotTS(axes[k+1], t, [predYC[:, 0, k], obsYC[:, k]],
                  styLst='-*', cLst='rb')
    axes[k+1].set_title(code+' '+codePdf.loc[code]['shortName'])
    axes[k].set_xticks([])
fig.show()

r = model.r.detach().cpu().numpy()
t = 1-np.arange(365)/365
fig, ax = plt.subplots(1, 1)
for k, code in enumerate(codeLst):
    c0 = r[0, k]
    c1 = r[1, k]
    rr = 10**r[2, k]
    gg = c0 * np.exp(-rr*t)*rr + c1*(1-np.exp(-rr*t))
    # gg = c1*(1-np.exp(-rr*t))
    ax.plot(365-np.arange(365), gg, label=codePdf.loc[code]['shortName'])
ax.legend()
ax.set_xlabel('lag day')
fig.show()


fig, ax = plt.subplots(1, 1)
t = dfY.index.values[365:]
tLst = ['2000-01-01', '2000-03-01', '2000-06-01', '2000-09-01']
for k, ts in enumerate(tLst):
    i = np.where(t == np.datetime64(ts))[0][0]
    gg = gate[i, 0, :].detach().cpu().numpy()
    b0 = b[i, 0, 0].detach().cpu().numpy()
    b1 = b[i, 0, 1].detach().cpu().numpy()
    ax.plot(365-np.arange(365), gg*b0, label='gate {}'.format(ts))
ax.legend()
ax.set_xlabel('lag day')
fig.show()

# recovery
p = x[:, 0, 0]
rho = 365
nt = len(p)
matQA = np.ndarray([nt-rho, rho])
for k in range(rho, nt):
    i = k-rho
    gg = gate[i, 0, :].detach().cpu().numpy()
    b0 = b[k, 0, 0].detach().cpu().numpy()
    b1 = b[k, 0, 1].detach().cpu().numpy()
    q = p[i:k]*gg*b0
    # qA = np.add.accumulate(q)
    qA = np.add.accumulate(q)+b1
    matQA[i, :] = qA
outQA = transform.transOut(matQA, mtdY[0], statY[0])

fig, ax = plt.subplots(1, 1, figsize=(16, 4))
t = dfY.index.values[rho:]
data = [y[rho:, 0, 0], matQA[:, -1],
        matQA[:, -60], matQA[:, -180], matQA[:, 0]]
axplot.plotTS(ax, t, data,
              legLst=['obs', '> 0 day', '> 60 day', '> 180 day', '= 365 day'], styLst='-----', cLst='kbcgm')
fig.show()
