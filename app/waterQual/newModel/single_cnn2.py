import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS, rnn, crit, cnn
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
import torch.nn as nn
import torch.nn.functional as F


siteNo = '02335870'

siteNo = '08195000'

code = '00955'
freq = 'W'
sn = 1

# load data
varF = gridMET.varLst+ntn.varLst
varC = usgs.varC
varQ = usgs.varQ
varLst = varF+varC+varQ
df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq='W')

# training / testing
yr = df.index.year.values
ind1 = np.where(yr <= 2016)[0]
ind2 = np.where(yr > 2016)[0]
dfYP = pd.DataFrame(index=df.index, columns=['WRTDS', 'CNN'])

# WRTDS
dfX = pd.DataFrame({'date': df.index}).set_index('date')
dfX = dfX.join(np.log(df['00060']+sn)).rename(
    columns={'00060': 'logQ'})
t = yr+dfX.index.dayofyear.values/365
dfX['sinT'] = np.sin(2*np.pi*t)
dfX['cosT'] = np.cos(2*np.pi*t)
x = dfX.iloc[ind1].values
y = df.iloc[ind1][code].values
[xx, yy], iv = utils.rmNan([x, y])
lrModel = LinearRegression()
lrModel = lrModel.fit(xx, yy)
b = dfX.isna().any(axis=1)
yp = lrModel.predict(dfX[~b].values)
dfYP.at[dfYP[~b].index, 'WRTDS'] = yp

# CNN
varC = [code]
rho = 52
dfX = pd.DataFrame({'date': df.index}).set_index('date')
dfX = dfX.join(np.log(df['pr']+sn))
dfY = pd.DataFrame({'date': df.index}).set_index('date')
dfY = dfY.join(np.log(df['00060']+sn))
dfXN = (dfX-dfX.min())/(dfX.max()-dfX.min())
dfYN = (dfY-dfY.min())/(dfY.max()-dfY.min())

xx = dfXN.values.swapaxes(1, 0)
xx = np.expand_dims(xx, 0)
yy = dfYN['00060'].values[rho-1:]
yy = np.expand_dims(yy, 0)

xT = torch.from_numpy(xx).float().cuda()
yT = torch.from_numpy(yy).float().cuda()

model = nn.Conv1d(1, 30, rho).cuda()
lossFun = crit.RmseLoss2D().cuda()
optim = torch.optim.Adadelta(model.parameters())

for iEp in range(50):
    t0 = time.time()
    yP = model(xT).mean(dim=1)
    loss = lossFun(yP, yT)
    loss.backward()
    optim.step()
    model.zero_grad()
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, loss, ct)
    print(logStr)
# model.zero_grad()

yPred = model(xT).mean(dim=1).detach().cpu().numpy().flatten()
yObs = dfYN['00060'].values[rho-1:]
t = dfYN.index[rho-1:]

# plot
fig, ax = plt.subplots(1, 1, figsize=(16, 6))
axplot.plotTS(ax, t, [yPred, yObs], styLst='--', cLst='rk')
fig.show()


pLst = list()
for p in model.parameters():
    pLst.append(p.data.detach().cpu().numpy())

fig, ax = plt.subplots(1, 1, figsize=(16, 6))
ax.plot(pLst[0][:, 0, :].T)
fig.show()
