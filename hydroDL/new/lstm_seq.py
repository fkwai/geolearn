import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS, rnn, crit
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from sklearn.linear_model import LinearRegression
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
import torch
import os
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

siteNo = '07060710'
code = '00955'
freq = 'D'
sn = 1

# load data
varF = gridMET.varLst+ntn.varLst
varC = usgs.varC
varQ = usgs.varQ
varLst = varF+varC+varQ
df = waterQuality.readSiteTS(siteNo, varLst=varLst)

# training / testing
yr = df.index.year.values
ind1 = np.where(yr <= 2005)[0]
ind2 = np.where(yr > 2005)[0]
dfYP = pd.DataFrame(index=df.index, columns=['WRTDS', 'LSTM'])

# LSTM
varC = ['00060', '00955']
rho = 365
dfX = pd.DataFrame({'date': df.index}).set_index('date')
# dfX = dfX.join(np.log(df['pr']+sn))
dfX = dfX.join(df['pr'])
dfXN = (dfX-dfX.quantile(0.1))/(dfX.quantile(0.9)-dfX.quantile(0.1))
dfY = pd.DataFrame({'date': df.index}).set_index('date')
# dfY = dfY.join(np.log(df['00060']+sn))
dfY = dfY.join(df['00060'])
dfY = dfY.join(df['00955'])
dfYN = (dfY-dfY.quantile(0.1))/(dfY.quantile(0.9)-dfY.quantile(0.1))
dfC = df[varC].dropna(how='any')
xLst = list()
yLst = list()
# orgnize data
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
    tempY = tempY.join(dfYN)
    yLst.append(tempY.values)
    tempX = temp.copy()
    tempX = tempX.join(dfXN)
    xLst.append(tempX.values)
x = np.stack(xLst, axis=-1).swapaxes(1, 2).astype(np.float32)
y = np.stack(yLst, axis=-1).swapaxes(1, 2).astype(np.float32)
x[np.where(np.isnan(x))] = -1
ind1 = dfC.index.year <= 2005
xx = x[:, ind1, :]
yy = y[:, ind1, :]
# training
nbatch = 100
nEp = 100
saveEp = [10, 50, 100]

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

    if iEp in saveEp:
        # testing
        xA = np.expand_dims(dfXN.values, axis=1)
        xA[np.where(np.isnan(xA))] = -1
        xF = torch.from_numpy(xA).float().cuda()
        yF = model(xF)
        dfYP['00060-ep'+str(iEp)] = yF[:, :,
                                       0].detach().cpu().numpy().flatten()
        dfYP['00955-ep'+str(iEp)] = yF[:, :,
                                       1].detach().cpu().numpy().flatten()

# plot data
t = df.index
yr = t.year.values
ind1 = (yr <= 2005) & (yr >= 1980)
ind2 = yr > 2005
o1 = df[ind1][code].values
o2 = df[ind2][code].values
t1 = t[ind1]
t2 = t[ind2]
ep = 100
v1 = [dfYP[ind1]['00060-ep'+str(ep)].values,
      dfYN[ind1]['00060'].values]
v2 = [dfYP[ind2]['00060-ep'+str(ep)].values,
      dfYN[ind2]['00060'].values]
v3 = [dfYP[ind1]['00955-ep'+str(ep)].values,
      dfYN[ind1]['00955'].values]
v4 = [dfYP[ind2]['00955-ep'+str(ep)].values,
      dfYN[ind2]['00955'].values]

# plot
fig, axes = plt.subplots(2, 1, figsize=(16, 6))
axplot.plotTS(axes[0], t1, v1, styLst='--*', cLst='rbk')
axplot.plotTS(axes[1], t2, v2, styLst='--*', cLst='rbk')
fig.show()

# plot
fig, axes = plt.subplots(2, 1, figsize=(16, 6))
axplot.plotTS(axes[0], t1, v3, styLst='-*', cLst='rbk')
axplot.plotTS(axes[1], t2, v4, styLst='-*', cLst='rbk')
fig.show()
