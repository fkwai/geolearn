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
from hydroDL.master import basinFull


# test for Q C seperate model
importlib.reload(trainBasin)
importlib.reload(test)
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

siteNo = '01184000'
# siteNoLst = [siteNo]
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
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
# varY = ['runoff']+codeSel
varY = codeSel
# mtdY = ['QT']
mtdY = dbBasin.io.extractVarMtd(varY)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)
trainSet = 'rmYr5'
testSet = 'pkYr5'


rho = 365
outName = dataName
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             nEpoch=100, batchSize=[rho, 200],
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
basinFull.trainModel(outName)

yPred, ycPred = basinFull.testModel(outName, DF=DF, testSet='all', reTest=True)

# PLOT
fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, DF.t, [yPred[:, 0, 0],DF.c[:,0,DF.varC.index('00915')]])
fig.show()