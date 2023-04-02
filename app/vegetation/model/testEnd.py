import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
from hydroDL import kPath
from hydroDL.model import rnn, crit, trainBasin
import torch
import time
from hydroDL.master import basinFull, slurm, dataTs2End

importlib.reload(hydroDL.data.dbVeg)
DF = dbVeg.DataFrameVeg('single')
trainName = '5fold_0_train'
testName = '5fold_0_test'

rho = 7
# subsetName='all'
DF.loadSubset(testName)
DM = dbVeg.DataModelVeg(DF, subset=testName)
outFolder = os.path.join(kPath.dirVeg, 'model', 'LSTMend', trainName)
DM.loadStat(outFolder)
dataTup = DM.getData()
dataEnd = dataTs2End(dataTup, rho)

dataEnd = trainBasin.dealNaN(dataEnd, [2, 2, 0, 0])
[nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup)


ep = 500
modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(ep))
model = rnn.LstmModel(nx + nxc, ny, 16, nLayer=3)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(modelStateFile))
else:
    model.load_state_dict(torch.load(modelStateFile, map_location=torch.device('cpu')))

# test
x = dataEnd[0]
xc = dataEnd[1]
ny = np.shape(dataTup[2])[2]
yOut, ycOut = trainBasin.testModel(model, x, xc, ny, batchSize=100)
yP = DM.transOutY(yOut)
yT = DM.transOutY(dataEnd[3])
a = yT
b = yP[-1, ...]
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.plot(a[:, 0], b[:, 0], '*')
fig.show()

from hydroDL import utils

rmse, corr = utils.stat.calErr(a[:, :, 0], b[:, :, 0])
corr
rmse
