import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, wqLinear, wqRela
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from hydroDL.model import rnn, crit
import os

siteNo = '401733105392404'
# siteNo = '01364959'
codeLst = ['00915', '00940', '00955']

varX = gridMET.varLst
varY = ['00060']
dfX = waterQuality.readSiteX(siteNo, varX)
dfY = waterQuality.readSiteY(siteNo, varY)

mtdX = waterQuality.extractVarMtd(varX)
mtdY = waterQuality.extractVarMtd(varY)
# mtdY[0] = 'norm'
# mtdX[0] = 'norm'
normX, statX = transform.transInAll(dfX.values, mtdX)
dfXN = pd.DataFrame(data=normX, index=dfX.index, columns=dfX.columns)
normY, statY = transform.transInAll(dfY.values, mtdY)
dfYN = pd.DataFrame(data=normY, index=dfY.index, columns=dfY.columns)

matX1 = dfXN[dfXN.index < np.datetime64('2000-01-01')].values
matY1 = dfYN[dfYN.index < np.datetime64('2000-01-01')].values
matX2 = dfXN[dfXN.index >= np.datetime64('2000-01-01')].values
matY2 = dfYN[dfYN.index >= np.datetime64('2000-01-01')].values
matX = dfXN.values
matY = dfYN.values

nx = len(varX)
ny = len(varY)
ind1 = np.where(~np.isnan(matY1))[0]
ind2 = np.where(~np.isnan(matY2))[0]
rho = 365
rhoF = 365
ind1 = ind1[ind1 > rho+rhoF]
nh = 512
ns = 50
importlib.reload(rnn)
model = rnn.modelTest(nx=nx, nh=nh, rho=rho).cuda()
model2 = rnn.CudnnLstmModel(nx=nx, ny=1, hiddenSize=nh).cuda()
optim = torch.optim.Adadelta(model.parameters())
optim2 = torch.optim.Adadelta(model2.parameters())
lossFun = crit.RmseLoss().cuda()
# train
model.train()
for i in range(50):
    t0 = time.time()
    x = np.ndarray([rho+rhoF, ns, nx])
    y = np.ndarray([rhoF, ns, ny])
    for k in range(ns):
        ind = ind1[np.random.randint(rho+rhoF, len(ind1))]
        x[:, k, :] = matX1[ind-rho-rhoF:ind, :]
        y[:, k, :] = matY1[ind-rhoF:ind, :]
    xx = torch.from_numpy(x).float().cuda()
    yy = torch.from_numpy(y).float().cuda()
    z = model(xx)
    loss = lossFun(z, yy)
    loss.backward()
    optim.step()
    model.zero_grad()
    print('{},{:.3f},{:.3f}'.format(i, loss, time.time()-t0))
    # print(z)

model2.train()
for i in range(50):
    t0 = time.time()
    x = np.ndarray([rho+rhoF, ns, nx])
    y = np.ndarray([rho+rhoF, ns, ny])
    for k in range(ns):
        ind = ind1[np.random.randint(rho+rhoF, len(ind1))]
        x[:, k, :] = matX1[ind-rho-rhoF:ind, :]
        y[:, k, :] = matY1[ind-rho-rhoF:ind, :]
    xx = torch.from_numpy(x).float().cuda()
    yy = torch.from_numpy(y).float().cuda()
    if i == 0:
        try:
            model2(xx)
        except:
            pass
    z = model2(xx)
    loss = lossFun(z, yy)
    loss.backward()
    optim2.step()
    model2.zero_grad()
    print('{},{:.3f},{:.3f}'.format(i, loss, time.time()-t0))

torch.cuda.empty_cache()

# test
model = model.train(mode=False).cpu()
model2 = model2.train(mode=False)

xx = torch.from_numpy(matX[:, None, :]).float()
xxCuda = xx.cuda()
z1, s, b = model(xx)
z2 = model2(xxCuda)

fig, ax = plt.subplots(1, 1)
ax.plot(z1.detach().cpu().numpy().flatten(), '-r')
ax.plot(z2.detach().cpu().numpy().flatten(), '-b')
ax.plot(matY, '-g')
fig.show()

saveDir = r'C:\Users\geofk\work\waterQuality\modelNew'
sFile = os.path.join(saveDir, '{}_S.csv'.format(siteNo))
for k in range(rho):
    k
    data = s[:, 0, k].cpu().detach().numpy().flatten()
    if k == 0:
        pd.DataFrame(data).T.to_csv(sFile, header=None, index=None)
    else:
        pd.DataFrame(data).T.to_csv(
            sFile, header=None, index=None, mode='a')
bFile = os.path.join(saveDir, '{}_B.csv'.format(siteNo))
for k in range(2):
    k
    data = b[:, 0, k].cpu().detach().numpy().flatten()
    if k == 0:
        pd.DataFrame(data).T.to_csv(bFile, header=None, index=None)
    else:
        pd.DataFrame(data).T.to_csv(
            bFile, header=None, index=None, mode='a')

zFile = os.path.join(saveDir, '{}_Z.csv'.format(siteNo))
data = z1.cpu().detach().numpy().flatten()
pd.DataFrame(data).T.to_csv(zFile, header=None, index=None)
lFile = os.path.join(saveDir, '{}_L.csv'.format(siteNo))
data = z2.cpu().detach().numpy().flatten()
pd.DataFrame(data).T.to_csv(lFile, header=None, index=None)
