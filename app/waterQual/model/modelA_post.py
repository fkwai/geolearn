import os
import time
import json
import numpy as np
import pandas as pd
import torch
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit
from hydroDL.post import plot
import matplotlib.pyplot as plt
from datetime import datetime as dt
from random import randint

# caseName = 'refBasins'
caseName = 'temp'
nEpoch = 100
modelFolder = os.path.join(kPath.dirWQ, 'modelA', caseName)
dictData, info, x, y, c = waterQuality.loadData(caseName)

targetFile = os.path.join(modelFolder, 'target.csv')
dfT = pd.read_csv(targetFile, dtype={'siteNo': str})
outFile = os.path.join(modelFolder, 'output_Ep' + str(nEpoch) + '.csv')
dfP = pd.read_csv(outFile, dtype={'siteNo': str})

siteNoLst = dictData['siteNoLst']
varC = dictData['varC']
iS = randint(0,len(siteNoLst))
iC = randint(0,21)
a = dfT[dfT['siteNo'] == siteNoLst[iS]]
b = dfP[dfP['siteNo'] == siteNoLst[iS]]
# t1 = pd.to_datetime(a[a['train'] == 1]['date'].values[-1])
# t2 = pd.to_datetime(a[a['train'] == 0]['date'].values[0])
# tBar = t1+(t2-t1)/2
var = varC[iC]
fig, ax = plt.subplots(1, 1)
fig, ax = plot.plotTS(t=a['date'], y=[a[var], b[var]], legLst=[
                      var+'_obs', var+'_pred'])
fig.show()
