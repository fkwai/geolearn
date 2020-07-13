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
nh = 256
batchSize = [365, 50]
if not waterQuality.exist(siteNo):
    wqData = waterQuality.DataModelWQ.new(siteNo, [siteNo])
wqData = waterQuality.DataModelWQ(siteNo)
wqData.c = wqData.c * wqData.q[-1, :, 0:1]

varX = wqData.varF
varXC = wqData.varG
varY = [wqData.varQ[0]]
varYC = ['00915', '00945', '00955']
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


model = rnn.LstmModel(
    nx=nx+nxc, ny=ny+nyc, hiddenSize=nh)
optim = torch.optim.Adadelta(model.parameters())
# lossFun = crit.RmseMix()
lossFun = crit.RmseLoss()
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()

# train
model.train()
model.zero_grad()
for k in range(200):
    t0 = time.time()
    xT, yT = trainTS.subsetRandom(dataTup, batchSize, sizeLst)
    if k == 0:
        try:
            yP = model(xT)
        except:
            print('first iteration failed again for CUDNN_STATUS_EXECUTION_FAILED ')
    yP = model(xT)
    # loss = lossFun(yP[:, :, :ny], yP[-1, :, ny:],
    #                yT[:, :, :ny], yT[-1, :, ny:])
    loss = lossFun(yP, yT)
    loss.backward()
    optim.step()
    model.zero_grad()
    print('{} {:.3f} {:.3f}'.format(k, loss, time.time()-t0))

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
yT = model(xT)
yO = yT.detach().cpu().numpy()


predY = transform.transOut(yO[:, :, 0], mtdY[0], statY[0])
predYC = transform.transOutAll(yO[:, :, 1:], mtdYC, statYC)/predY
obsY = dfY.values
obsYC = dfYC.values

t = dfY.index.values
fig, axes = plt.subplots(4, 1)
axplot.plotTS(axes[0], t, [predY, obsY],
              styLst='---', cLst='rb')
axes[0].set_title('streamflow')
axes[0].set_xticks([])
codePdf = usgs.codePdf
for k, code in enumerate(varYC):
    axplot.plotTS(axes[k+1], t, [predYC[:, 0, k], obsYC[:, k]],
                  styLst='-*', cLst='rb')
    axes[k+1].set_title(code+' '+codePdf.loc[code]['shortName'])
    axes[k].set_xticks([])
fig.show()