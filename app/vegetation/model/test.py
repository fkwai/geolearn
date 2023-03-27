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
from hydroDL.master import basinFull, slurm

importlib.reload(hydroDL.data.dbVeg)
DF = dbVeg.DataFrameVeg('single')
trainName = '5fold_0_train'
testName = '5fold_0_test'


# subsetName='all'
DF.loadSubset(testName)
DM = dbVeg.DataModelVeg(DF, subset=testName)
outFolder = os.path.join(kPath.dirVeg, 'model', 'LSTMfull', trainName)
DM.loadStat(outFolder)
dataTup = DM.getData()
dataTup = trainBasin.dealNaN(dataTup, [1, 1, 0, 0])
[nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup)

ep = 200
modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(ep))
model = rnn.LstmModel(nx + nxc, ny, 16, nLayer=3)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(modelStateFile))
else:
    model.load_state_dict(torch.load(modelStateFile, map_location=torch.device('cpu')))

# test
x = dataTup[0]
xc = dataTup[1]
ny = np.shape(dataTup[2])[2]
yOut, ycOut = trainBasin.testModel(model, x, xc, ny, batchSize=100)
yP = DM.transOutY(yOut)
a = DM.Y
b = yP
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.plot(a[:, :, 0], b[:, :, 0], '*')
fig.show()

from hydroDL import utils

rmse, corr = utils.stat.calErr(a[:, :, 0], b[:, :, 0])
corr
rmse
