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

# load old data
outFile = '/home/kuai/work/VegetationWater/data/model/data/testData.npz'
dataOld = np.load(outFile)
xTrain = dataOld['xTrain'].swapaxes(0, 1)
xTest = dataOld['xTest'].swapaxes(0, 1)
yTrain = dataOld['yTrain'][:, None]
yTest = dataOld['yTest'][:, None]
dataTup1 = (xTrain, None, None, yTrain)
dataTup2 = (xTest, None, None, yTest)
outFolder = os.path.join(kPath.dirVeg, 'model', 'old')
[nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup1)
rho = 7

ep = 500
modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(ep))
model = rnn.LstmModel(nx + nxc, ny+nyc, 10, nLayer=4)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(modelStateFile))
else:
    model.load_state_dict(torch.load(modelStateFile, map_location=torch.device('cpu')))

# test
x = dataTup2[0]
xc = dataTup2[1]
ny = np.shape(dataTup2[3])[2]
yOut, ycOut = trainBasin.testModel(model, x, xc, ny+nyc, batchSize=100)
yP = yOut
yT = dataTup2[3]
a = yT
b = yP[-1, ...]
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.plot(a[:, 0], b[:, 0], '*')
fig.show()

from hydroDL import utils

rmse, corr = utils.stat.calErr(a[:, 0], b[:, 0])
corr
rmse
