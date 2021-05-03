import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS, rnn, crit
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin

siteNo = '06800500'
codeLst = ['00915', '00925']

df = GLASS.readBasin(siteNo)

# plot data
df = dbBasin.readSiteTS(siteNo, varLst=codeLst, freq='D')
nc = len(codeLst)
fig, axes = plt.subplots(nc, 1, figsize=(12, 3))
for k, code in enumerate(codeLst):
    axplot.plotTS(axes[k], df.index, df[code].values, cLst='k')
fig.show()


# load data
sd = '1982-01-01'
ed = '2018-12-31'
siteNoLst = [siteNo]
dm = dbBasin.DataModelFull.new('test', siteNoLst, sdStr=sd, edStr=ed)

# load data
varF = gridMET.varLst+ntn.varLst
varC = usgs.varC
varQ = usgs.varQ
varLst = varF+varC+varQ
varLst = GLASS.varLst
df = dbBasin.readSiteTS(siteNo, varLst=varLst, freq='W')

# training / testing
yr = df.index.year.values
ind1 = np.where(yr <= 2016)[0]
ind2 = np.where(yr > 2016)[0]
dfYP = pd.DataFrame(index=df.index, columns=['WRTDS', 'LSTM'])


# LSTM
varC = [code]
rho = 52
dfX = pd.DataFrame({'date': df.index}).set_index('date')
dfX = dfX.join(np.log(df['00060']+sn)).rename(
    columns={'00060': 'logQ'})
t = yr+dfX.index.dayofyear.values/365
dfX['sinT'] = np.sin(2*np.pi*t)
dfX['cosT'] = np.cos(2*np.pi*t)
# dfX['T'] = (dfX.index.values-np.datetime64('1979-01-01', 'D')
#       ).astype('timedelta64[D]')
# dfX = dfX.join(df[varF])
dfXN = (dfX-dfX.min())/(dfX.max()-dfX.min())
dfC = df[varC].dropna(how='all')
xLst = list()
yLst = list()
# reorgnize data
for k in range(len(dfC)):
    ct = dfC.index[k]
    if freq == 'D':
        ctR = pd.date_range(
            ct-pd.Timedelta(days=rho-1), ct)
    elif freq == 'W':
        ctR = pd.date_range(
            ct-pd.Timedelta(days=rho*7-1), ct, freq='W-TUE')
    temp = pd.DataFrame({'date': ctR}).set_index('date')
    tempY = temp.copy()
    tempY = tempY.join(df[varC])
    yLst.append(tempY.values)
    tempX = temp.copy()
    tempX = tempX.join(dfXN)
    xLst.append(tempX.values)
x = np.stack(xLst, axis=-1).swapaxes(1, 2).astype(np.float32)
y = np.stack(yLst, axis=-1).swapaxes(1, 2).astype(np.float32)
x[np.where(np.isnan(x))] = -1
ind1 = dfC.index.year <= 2016
xx = x[:, ind1, :]
yy = y[:, ind1, :]
# training
nbatch = 20
nEp = 100
ns = xx.shape[1]
nx = xx.shape[-1]
ny = yy.shape[-1]
model = rnn.LstmModel(nx=nx, ny=ny, hiddenSize=256).cuda()
optim = torch.optim.Adadelta(model.parameters())
lossFun = crit.RmseLoss().cuda()
nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - nbatch / ns)))
lossLst = list()
for iEp in range(1, nEp + 1):
    lossEp = 0
    t0 = time.time()
    for iIter in range(nIterEp):
        iR = np.random.randint(0, ns, nbatch)
        xTemp = xx[:, iR, :]
        yTemp = yy[:, iR, :]
        xT = torch.from_numpy(xTemp).float().cuda()
        yT = torch.from_numpy(yTemp).float().cuda()
        if iEp == 1 and iIter == 0:
            try:
                yP = model(xT)
            except:
                print('first iteration failed again for CUDNN_STATUS_EXECUTION_FAILED ')
        yP = model(xT)
        loss = lossFun(yP, yT)
        loss.backward()
        optim.step()
        model.zero_grad()
        lossEp = lossEp + loss.item()
    lossEp = lossEp / nIterEp
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, lossEp, ct)
    print(logStr)
    lossLst.append(loss)

# testing
xA = np.expand_dims(dfXN.values, axis=1)
xA[np.where(np.isnan(xA))] = -1
xF = torch.from_numpy(xA).float().cuda()
yF = model(xF)
dfYP['LSTM'] = yF.detach().cpu().numpy().flatten()

# plot data
fig, ax = plt.subplots(1, 1, figsize=(16, 3))
ax.plot(df.index, df[code].values, 'k*')
ax.plot(dfYP.index, dfYP['WRTDS'].values, '-b')
ax.plot(dfYP.index, dfYP['LSTM'].values, '-r')
fig.show()
