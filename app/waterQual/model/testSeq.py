from hydroDL.post import axplot
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
# master = basins.loadMaster('HBN-opt2')
# wqData = waterQuality.DataModelWQ(master['dataName'])
# p1, o1 = basins.testModel('HBN-first50-opt2', 'first50', wqData=wqData)


outName = 'HBN-first80-opt2'
# testset = 'first50'
master = basins.loadMaster(outName)
statTup = basins.loadStat(outName)
model = basins.loadModel(outName)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()


# load test data
wqData = waterQuality.DataModelWQ(master['dataName'])
(varX, varXC) = (master['varX'], master['varXC'])
(varY, varYC) = (master['varY'], master['varYC'])
statX, statXC, statY, statYC = statTup

siteNoLst = wqData.info['siteNo'].unique().tolist()

nFill = 5
startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2019, 12, 31)

tabG = gageII.readData(varLst=varXC, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)

outLst = list()
for i, siteNo in enumerate(siteNoLst):
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
    xA = dfX.values

    nt = len(dfX)
    ny = len(varY) if varY is not None else 0
    nyc = len(varYC) if varYC is not None else 0
    out = np.full([nt, ny+nyc], np.nan)

    mtdX = wqData.extractVarMtd(varX)
    x = transform.transInAll(xA, mtdX, statLst=statX)
    mtdXC = wqData.extractVarMtd(varXC)
    xc = transform.transInAll(tabG.loc[siteNo].values, mtdXC, statLst=statXC)
    tN = dfX.index[dfX.isna().any(axis=1)].values.astype('datetime64[D]')

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
            [x[ind1:ind2, :], np.tile(xc, [ind2-ind1, 1])], axis=-1)
        xx = np.expand_dims(xx, axis=0)
        xT = torch.from_numpy(xx).float()
        if torch.cuda.is_available():
            xT = xT.cuda()
        if i == 0 and ind1 == 0:
            try:
                yT = model(xT)
            except:
                print('first iteration failed again')
        yT = model(xT)
        out[ind1:ind2, :] = yT.detach().cpu().numpy()
    outLst.append(out)
    print('tested site {} cost {:.3f}'.format(i, time.time()-t0))
# if i == 0:
#     try:
#         yT = model(xT)
#     except:
#         print('first iteration failed again')

iS = 10
siteNo = siteNoLst[iS]
q = outLst[iS][:, 0]

mtdY = wqData.extractVarMtd(varY)
statY = statTup[2][0]
qP = np.exp(q*(statY[1]-statY[0])+statY[0])-1

# qP = wqData.transOut(q, statTup[2], varY)
dfF = gridMET.readBasin(siteNo)
dfX = dfF
dfX = dfX[dfX.index >= startDate]
dfX = dfX[dfX.index <= endDate]
dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
dfY = dfX.join(dfQ)
qT = dfY['00060_00003'].values
qX = (np.log(qT+1)-statY[0])/(statY[1]-statY[0])
t = dfY.index.values.astype('datetime64[D]')

fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, t, [qX, q], legLst=['obs', 'pred'])
fig.show()



iS = 10
siteNo = siteNoLst[iS]
q1 = outLst[10][:, 0]
q2 = outLst[20][:, 0]

fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, t, [q1, q2], legLst=['obs', 'pred'])
fig.show()
