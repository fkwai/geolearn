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

siteNo = '01434025'
codeLst = ['00915', '00940', '00955']
# codeLst = ['00915', '00955']
nh = 256
batchSize = [365, 50]
if not waterQuality.exist(siteNo):
    wqData = waterQuality.DataModelWQ.new(siteNo, [siteNo])
wqData = waterQuality.DataModelWQ(siteNo)
varX = wqData.varF
varXC = wqData.varG
varY = [wqData.varQ[0]]
varYC = codeLst
varTup = (varX, varXC, varY, varYC)
dataTup, statTup = wqData.transIn(varTup=varTup)

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
t = dfY.index.values

fig, axes = plt.subplots(2, 1)
axplot.plotTS(axes[0], t, [x[:, 0, 0], y[:, 0, 0]],
              styLst='---', cLst='rgb')

axplot.plotTS(axes[1], t, [yc[:, 0, k] for k in range(len(varYC))],
              styLst='***', cLst='rgb')
fig.show()
