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
import pandas as pd

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
if not os.path.exists(outFolder):
    os.mkdir(outFolder)

[nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup1)


# define model, loss, optim
lossFun = crit.RmseLoss2D()
model = rnn.LstmModel(nx + nxc, ny+nyc, 10, nLayer=4)
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()
optim = torch.optim.NAdam(model.parameters())

# train
nEp = 500
sEp = 50
resumeEpoch = 0
logFile = os.path.join(outFolder, 'log')
if os.path.exists(logFile) and resumeEpoch == 0:
    os.remove(logFile)
logH = open(logFile, 'a')
if resumeEpoch > 0:
    timeStr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('resume {} Ep{}'.format(timeStr, resumeEpoch), logH, flush=True)
    print('resume {} Ep{}'.format(timeStr, resumeEpoch), flush=True)
for k in range(resumeEpoch, nEp, sEp):
    model, optim = trainBasin.trainModel(
        dataTup1,
        model,
        lossFun,
        optim,
        batchSize=[7, 20000],
        nEp=sEp,
        cEp=k,
        outFolder=outFolder,
        logH=logH,
        optBatch='Index',
    )
    # save model
    modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(k + sEp))
    torch.save(model.state_dict(), modelStateFile)
