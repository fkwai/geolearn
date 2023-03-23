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
subsetName = '5fold_0_train'


# subsetName='all'
DF.loadSubset(subsetName)
DM = dbVeg.DataModelVeg(DF, subsetName=subsetName)
outFolder = os.path.join(kPath.dirVeg, 'model', 'LSTMfull')
DM.trans(
    mtdX=['norm' for x in DM.varX],
    mtdXC=['norm' for x in DM.varXC],
    mtdY=['norm' for x in DM.varY],
    mtdYC=None,
)
DM.saveStat(outFolder)
dataTup = DM.getData()
dataTup = trainBasin.dealNaN(dataTup, [1, 1, 0, 0])
[nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup)

# define model, loss, optim
lossFun = crit.MSELoss()
model = rnn.LstmModel(nx, ny, 128, nLayer=2)
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

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
        dataTup,
        model,
        lossFun,
        optim,
        batchSize=[50, 100],
        nEp=sEp,
        cEp=k,
        outFolder=outFolder,
        logH=logH,
    )
    # save model
    modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(k+sEp))
    torch.save(model.state_dict(), modelStateFile)
