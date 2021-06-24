import pandas as pd
from hydroDL.data import dbBasin
import numpy as np
import matplotlib.pyplot as plt
import os
from hydroDL.master import basinFull
import torch
from hydroDL.model import rnn, crit, trainBasin

outName = 'weathering-FPR2QC-t365-B10'
ep = 100

# save
outFolder = basinFull.nameFolder(outName)
modelFile = os.path.join(outFolder, 'model_ep{}'.format(ep))
model = torch.load(modelFile)
modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(ep))
torch.save(model.state_dict(), modelStateFile)

# load
dictP = basinFull.loadMaster(outName)
DF = dbBasin.DataFrameBasin(dictP['dataName'])
dictVar = {k: dictP[k]
           for k in ('varX', 'varXC', 'varY', 'varYC')}
DM = dbBasin.DataModelBasin(DF, subset='A10', **dictVar)
DM.loadStat(outFolder)
dataTup = DM.getData()
[nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup)
dataTup = trainBasin.dealNaN(dataTup, dictP['optNaN'])
if dictP['modelName'] == 'CudnnLSTM':
    model = rnn.CudnnLstmModel(
        nx=nx+nxc, ny=ny+nyc, hiddenSize=dictP['hiddenSize'])
elif dictP['modelName'] == 'LstmModel':
    model = rnn.LstmModel(
        nx=nx+nxc, ny=ny+nyc, hiddenSize=dictP['hiddenSize'])
else:
    raise RuntimeError('Model not specified')
model.load_state_dict(torch.load(modelStateFile))
