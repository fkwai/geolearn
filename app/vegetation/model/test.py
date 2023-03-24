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
DM = dbVeg.DataModelVeg(DF, subsetName=testName)
outFolder = os.path.join(kPath.dirVeg, 'model', 'LSTMfull')
DM.loadStat(outFolder)

dataTup = DM.getData()
dataTup = trainBasin.dealNaN(dataTup, [1, 1, 0, 0])
[nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup)

# define model, loss, optim
lossFun = crit.MSELoss()
model = rnn.LstmModel(nx + nxc, ny, 128, nLayer=2)
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# test
x = dataTup[0]
xc = dataTup[1]
ny = np.shape(dataTup[2])[2]
yOut, ycOut = trainBasin.testModel(model, x, xc, ny, batchSize=100)
a=DM.Y
b=yOut
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,1)
ax.plot(a[:,:,0],b[:,:,0],'*')
fig.show()

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
    modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(k + sEp))
    torch.save(model.state_dict(), modelStateFile)
