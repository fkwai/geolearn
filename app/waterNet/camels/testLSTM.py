

import torch
import os
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
from hydroDL.model import trainBasin
import numpy as np

dataName = 'camelsM'
trainSet = 'B05'
testSet = 'A05'
dataLst = ['camelsN', 'camelsD', 'camelsM']

# LSTM
DF = dbBasin.DataFrameBasin(dataName)
outName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=True, ep=100)

Q = DF.extractSubset(DF.q, subsetName=testSet)
y = Q[:, :, 1]
k = 100
fig, ax = plt.subplots(1, 1)
ax.plot(yL[:, k, 0], 'b')
ax.plot(y[:, k], 'k')
fig.show()

dictP = basinFull.loadMaster(outName)
outFolder = basinFull.nameFolder(outName)
dictVar = {k: dictP[k] for k in ('varX', 'varXC', 'varY', 'varYC')}
DM = dbBasin.DataModelBasin(DF, subset=testSet, **dictVar)
DM.loadStat(outFolder)
dataTup = DM.getData()
dataTup = trainBasin.dealNaN(dataTup, dictP['optNaN'])

np.where(np.isnan(dataTup[0]))

dataTup[1].shape

x = dataTup[0]
xc = dataTup[1]
y = dataTup[2]

model = basinFull.defineModel(dataTup, dictP)
model = basinFull.loadModelState(outName, 100, model)

yOut, ycOut = trainBasin.testModel(model, x, xc, 1, batchSize=10)

modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(500))
mm = torch.load(modelStateFile)
for k in mm.keys():
    torch.where(torch.isnan(mm[k]))


fig, ax = plt.subplots(1, 1)
# ax.plot(yL[:, k, 0], 'b')
ax.plot(y[:, 0,0], 'k')
fig.show()
