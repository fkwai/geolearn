import importlib
from hydroDL.master import basins
from hydroDL.app.waterQuality import WRTDS
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
from hydroDL.model import rnn, crit, trainBasin, test
import torch
from torch import nn

# test for Q C seperate model

importlib.reload(test)
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

siteNo = '01184000'
siteNoLst = [siteNo]
sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'test'
freq = 'D'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)

# create subset
yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(DF.t, yrIn)
t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
DF.createSubset('pkYr5', dateLst=t1)
DF.createSubset('rmYr5', dateLst=t2)

# define inputs
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']

varX = gridMET.varLst+ntn.varLst
# varX = ['pr']
mtdX = dbBasin.io.extractVarMtd(varX)
varXC = None
mtdXC = dbBasin.io.extractVarMtd(varXC)
# varY = ['runoff']+codeSel
varY = codeSel
# mtdY = ['QT']
mtdY = dbBasin.io.extractVarMtd(varY)

varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)
trainSet = 'rmYr5'
testSet = 'pkYr5'

d1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varY=varY, varXC=varXC, varYC=varYC)
d1.trans(mtdX=mtdX, mtdXC=mtdXC, mtdY=mtdY, mtdYC=mtdYC)
dataLst = d1.getData()
dataLst = trainBasin.dealNaN(dataLst, [1, 1, 0, 0])

# train
importlib.reload(test)
sizeLst = trainBasin.getSize(dataLst)
[nx, nxc, ny, nyc, nt, ns] = sizeLst
model = test.LSTM(nx+nxc, ny+nyc, 256).cuda()
lossFun = crit.RmseLoss().cuda()
optim = torch.optim.Adadelta(model.parameters())

rho = 365
nbatch = 100
nEp = 100
matB = ~np.isnan(dataLst[2][rho:, :, :])
nD = np.sum(np.any(matB, axis=2))
if nbatch*rho > nD:
    nIterEp = 1
else:
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - nbatch*rho/nD)))

# TRAIN
lossEp = 0
cEp = 0
lossEpLst = list()
t0 = time.time()
model.cuda()
model.train()
model.zero_grad()
for iEp in range(1, nEp + 1):
    lossEp = 0
    t0 = time.time()
    # somehow the first iteration always failed
    if iEp == 1:
        try:
            xT, yT = trainBasin.subsetRandom(dataLst, [rho, nbatch],
                                             sizeLst, opt='Weight')
            yP = model(xT)
        except:
            print('first iteration failed again for CUDNN_STATUS_EXECUTION_FAILED ')
    for iIter in range(nIterEp):
        xT, yT = trainBasin.subsetRandom(dataLst, [rho, nbatch],
                                         sizeLst, opt='Weight', matB=matB)
        # yP1, yP = model(xT[:, :, 1:], xT[:, :, 0:1])
        yP = model(xT)
        if type(lossFun) is crit.RmseLoss2D:
            loss = lossFun(yP, yT[-1, :, :])
        else:
            loss = lossFun(yP, yT)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        # model.zero_grad()
        lossEp = lossEp + loss.item()
        # except:
        #     print('iteration Failed: iter {} ep {}'.format(iIter, iEp+cEp))
    lossEp = lossEp / nIterEp
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp+cEp, lossEp, ct)
    print(logStr, flush=True)
    # log.write(logStr+'\n')
    lossEpLst.append(lossEp)

# TEST
d2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varY=varY, varXC=varXC, varYC=varYC)
d2.borrowStat(d1)
dataTest = d2.getData()
dataTest = trainBasin.dealNaN(dataTest, [1, 1, 0, 0])
xTest = torch.from_numpy(dataLst[0]).float().cuda()
yTest = model(xTest)
# yTest = model(xTest)
yOut = yTest.detach().cpu().numpy()
yPred = d2.transOutY(yOut)


# PLOT
fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, d2.t, [yOut[:, 0, 0], d2.y[:, 0, 0]])
fig.show()
fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, d2.t, [yOut[:, 0, 1], d2.y[:, 0, 1]])
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(yT[:, 0, 0].detach().cpu().numpy(), '*r')
ax.plot(yP[:, 0, 0].detach().cpu().numpy(), '-b')
fig.show()


k = 0
# dataPlot = [yP[:, k, :], d1.Y[:, k, :], d2.Y[:, k, :]]
dataPlot = [yOut[:, k, :], d1.y[:, k, :], d2.y[:, k, :]]
cLst = ['red', 'grey', 'black']
fig, axes = figplot.multiTS(
    DF.t, dataPlot, cLst=cLst)
fig.show()

for k in range(len(varY)):
    utils.stat.calCorr(yOut[:, 0, k], d2.y[:, 0, k])


w = model.linearOut._parameters['weight'].detach().cpu().numpy()
b = model.linearOut._parameters['bias'].detach().cpu().numpy()
