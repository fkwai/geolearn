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
dfYP = pd.DataFrame(index=df.index, columns=['WRTDS', 'LSTM'])

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

# LSTM
varC = [code]
rho = 52
dfX = pd.DataFrame({'date': df.index}).set_index('date')
dfX = dfX.join(np.log(df['00060']+sn)).rename(
    columns={'00060': 'logQ'})
t = yr+dfX.index.dayofyear.values/365
dfX['sinT'] = np.sin(2*np.pi*t)
dfX['cosT'] = np.cos(2*np.pi*t)
dfX['T'] = (dfX.index.values-np.datetime64('1979-01-01', 'D')
      ).astype('timedelta64[D]')
dfX = dfX.join(df[varF])
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
model = rnn.LstmModel(nx=x.shape[-1], ny=1, hiddenSize=256).cuda()
optim = torch.optim.Adadelta(model.parameters())
lossFun = crit.RmseLoss().cuda()
nbatch = 20
t0 = time.time()
for k in range(100):
    ns = xx.shape[1]
    iR = np.random.randint(0, ns, nbatch)
    xTemp = xx[:, iR, :]
    yTemp = yy[:, iR, :]
    xT = torch.from_numpy(xTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    if k == 0:
        try:
            yP = model(xT)
        except:
            print('first iteration failed again for CUDNN_STATUS_EXECUTION_FAILED ')
    yP = model(xT)
    loss = lossFun(yP, yT)
    loss.backward()
    optim.step()
    model.zero_grad()
    print('{} {:.3f} {:.3f}'.format(k, loss, time.time()-t0))
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
