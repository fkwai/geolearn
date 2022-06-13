
import matplotlib
import matplotlib.gridspec as gridspec
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from hydroDL import utils
import os
from hydroDL.model import trainBasin, crit, waterNetTest
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
import pandas as pd
from hydroDL.model import waterNetTest, waterNet
from hydroDL.master import basinFull
import importlib

importlib.reload(waterNetTest)
importlib.reload(crit)

dataName = 'QN90ref'
# dataName = 'temp'
DF = dbBasin.DataFrameBasin(dataName)
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()

# model
nh = 16
ng = len(varXC)
ns = len(DF.siteNoLst)
nr = 5
model = waterNet.WaterNet0119(nh, len(varXC), nr)
model = model.cuda()

# water net
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'wn0119-{}-ep{}'.format('QN90ref', 300)
model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))
model.eval()
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
nt, ns, _ = y.shape
t = DF.getT(testSet)
testBatch = 100
iS = np.arange(0, ns, testBatch)
iE = np.append(iS[1:], ns)
yP = np.ndarray([nt-nr+1, ns])
for k in range(len(iS)):
    print('batch {}'.format(k))
    yOut = model(xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]])
    yP[:, iS[k]:iE[k]] = yOut.detach().cpu().numpy()
model.zero_grad()


# LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=False, ep=1000)
yL = yL[:, :, 0]

nash1 = utils.stat.calNash(yP, y[nr-1:, :, 0])
corr1 = utils.stat.calCorr(yP, y[nr-1:, :, 0])
nash2 = utils.stat.calNash(yL, y[:, :, 0])
corr2 = utils.stat.calCorr(yL, y[:, :, 0])

lat, lon = DF.getGeo()

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})

# map
figM = plt.figure(figsize=(10, 10))
gsM = gridspec.GridSpec(2, 1)
axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash1, vRange=[0, 1])
axM0.set_title('waterNet NSE')
axM1 = mapplot.mapPoint(figM, gsM[1, 0], lat, lon, nash2, vRange=[0, 1])
axM1.set_title('LSTM NSE')
dirPaper = r'C:\Users\geofk\work\waterQuality\paper\waterNet'
figM.show()
figM.savefig(os.path.join(dirPaper, 'mapNash'))

# box
fig, axes = figplot.boxPlot([[nash1, nash2], [corr1, corr2]],
                            label1=['NSE', 'corr'],
                            label2=['waterNet', 'LSTM'],
                            yRange=[0, 1], figsize=(6, 4))
fig.show()
fig.savefig(os.path.join(dirPaper, 'box'))

