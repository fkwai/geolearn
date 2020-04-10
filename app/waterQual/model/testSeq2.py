from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# training
dataName = 'temp10'
wqData = waterQuality.DataModelWQ(dataName)
masterName = basins.wrapMaster(
    dataName=dataName, trainName='first80', batchSize=[
        None, 100], outName='temp10-Q', varYC=None, nEpoch=100)
basins.trainModelTS(masterName)


outName = 'temp10-Q'
dictP = basins.loadMaster(outName)
statTup = basins.loadStat(outName)
model = basins.loadModel(outName)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
siteNoLst = wqData.info['siteNo'].unique().tolist()

(varX, varXC, varY, varYC) = (
    dictP['varX'], dictP['varXC'], dictP['varY'], dictP['varYC'])
statX, statXC, statY, statYC = statTup
startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2019, 12, 31)
nFill = 5

# testing using training data
dataTup, statTup = wqData.transIn(varTup=(varX, varXC, varY, varYC))
dataTup = trainTS.dealNaN(dataTup, dictP['optNaN'])

info = wqData.info
siteNo = siteNoLst[0]
ind = info[info['siteNo'] == siteNo].index

k = 0
x = dataTup[0][:, k, :]
xc = dataTup[1][k, :]
xx = np.concatenate([x, np.tile(xc, [365, 1])], axis=-1)
xx = np.expand_dims(xx, axis=0)
xT = torch.from_numpy(xx).float()
if torch.cuda.is_available():
    xT = xT.cuda()
yT = model(xT)
out = yT.detach().cpu().numpy()
obs = dataTup[2][:, k, :]
t2 = np.datetime64(info.iloc[4]['date'], 'D')
t1 = t2-np.timedelta64(365, 'D')
t = np.arange(t1, t2)

fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, t, [out.flatten(), obs.flatten()])
fig.show()

tabG = gageII.readData(varLst=varXC, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)

siteNo = siteNoLst[0]
t0 = time.time()
dfF = gridMET.readBasin(siteNo)
if '00060' in varX:
    dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
    dfQ = dfQ.rename(columns={'00060_00003': '00060'})
    dfX = dfQ.join(dfF)
else:
    dfX = dfF
dfX = dfX[dfX.index >= startDate]
dfX = dfX[dfX.index <= endDate]
dfX = dfX.interpolate(limit=nFill, limit_direction='both')
xA = np.expand_dims(dfX.values, axis=0)
xcA = np.expand_dims(tabG.loc[siteNo].values.astype(np.float), axis=0)

mtdX = wqData.extractVarMtd(varX)
x = transform.transInAll(xA, mtdX, statLst=statX)
mtdXC = wqData.extractVarMtd(varXC)
xc = transform.transInAll(xcA, mtdXC, statLst=statXC)

tN = dfX.index[dfX.isna().any(axis=1)].values.astype('datetime64[D]')

nt = len(dfX)
ny = len(varY) if varY is not None else 0
nyc = len(varYC) if varYC is not None else 0
out = np.full([nt, ny+nyc], np.nan)

x, xc = trainTS.dealNaN((x, xc), master['optNaN'][:2])
if len(tN) == 0:
    indLst1 = [0]
    indLst2 = [len(dfX)]
else:
    tA = dfX.index.values.astype('datetime64[D]')
    temp = tN[1:]-tN[:-1]
    t1Ary = tN[:-1][temp > np.timedelta64(1, 'D')]
    t2Ary = tN[1:][temp > np.timedelta64(1, 'D')]
    indLst1 = [0] +\
        np.where(tA == t1Ary)[0].tolist() + \
        list(np.where(tA == tN[-1])[0]+1)
    indLst2 = np.where(tA == tN[0])[0].tolist() + \
        np.where(tA == t2Ary)[0].tolist() +\
        [len(dfX)]

for ind1, ind2 in zip(indLst1, indLst2):
    if ind1 == ind2:
        break
    xx = np.concatenate(
        [x[ind1:ind2, :], np.tile(xc[0, :], [1, ind2-ind1, 1])], axis=-1)
    # xx = np.expand_dims(xx, axis=0)
    xT = torch.from_numpy(xx).float()
    if torch.cuda.is_available():
        xT = xT.cuda()
    # if i == 0 and ind1 == 0:
    #     try:
    #         yT = model(xT)
    #     except:
    #         print('first iteration failed again')
    yT = model(xT)
    out[ind1:ind2, :] = yT.detach().cpu().numpy()
# outLst.append(out)
# print('tested site {} cost {:.3f}'.format(i, time.time()-t0))


dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
dfQ = dfQ.rename(columns={'00060_00003': '00060'})
dfQ = dfQ[dfQ.index >= startDate]
dfQ = dfQ[dfQ.index <= endDate]
qT = dfQ['00060'].values

mtdQ = wqData.extractVarMtd(['00060'])
qN = transform.transInAll(qT, mtdQ, statLst=statY)

q = out[:, 0]

fig, ax = plt.subplots(1, 1)
ax.plot(q, qT, '*')
fig.show()
