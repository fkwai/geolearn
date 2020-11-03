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
# t = yr+dfX.index.dayofyear.values/365
# dfX['sinT'] = np.sin(2*np.pi*t)
# dfX['cosT'] = np.cos(2*np.pi*t)
# dfX['T'] = (dfX.index.values-np.datetime64('1979-01-01', 'D')
#       ).astype('timedelta64[D]')
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
nbatch = 20
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
        dfYP['LSTM-ep'+str(iEp)] = yF.detach().cpu().numpy().flatten()

# plot data
t = df.index
yr = t.year.values
ind1 = (yr <= 2016) & (yr >= 1980)
ind2 = yr > 2016
o1 = df[ind1][code].values
o2 = df[ind2][code].values
t1 = t[ind1]
t2 = t[ind2]
ep = 100
v1 = [dfYP[ind1]['LSTM-ep'+str(ep)].values,
      dfYP[ind1]['WRTDS'].values.astype(float),
      df[ind1][code].values]
v2 = [dfYP[ind2]['LSTM-ep'+str(ep)].values,
      dfYP[ind2]['WRTDS'].values.astype(float),
      df[ind2][code].values]
rmseWRTDS1, corrWRTDS1 = utils.stat.calErr(v1[1], v1[2])
rmseLSTM1, corrLSTM1 = utils.stat.calErr(v1[0], v1[2])
rmseWRTDS2, corrWRTDS2 = utils.stat.calErr(v2[1], v2[2])
rmseLSTM2, corrLSTM2 = utils.stat.calErr(v2[0], v2[2])

# plot
fig, axes = plt.subplots(2, 1, figsize=(16, 6))
axplot.plotTS(axes[0], t1, v1, styLst='--*', cLst='rbk')
axplot.plotTS(axes[1], t2, v2, styLst='--*', cLst='rbk')
axes[0].set_title('site {} WRTDS {:.2f} LSTM {:.2f}'.format(
    siteNo, corrWRTDS1, corrLSTM1))
axes[1].set_title('site {} WRTDS {:.2f} LSTM {:.2f}'.format(
    siteNo, corrWRTDS2, corrLSTM2))
# fig.show()
# plot forcing
# ax1 = axes[0].twinx()
# ax2 = axes[1].twinx()
# var = 'etr'
# axplot.plotTS(ax1, t1, df[ind1][var].values, styLst=['-.'], cLst='c')
# axplot.plotTS(ax2, t2, df[ind2][var].values, styLst=['-.'], cLst='c')
fig.show()
varF
