
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
trainSet = 'WYB09'
testSet = 'WYA09'
# dataName = 'QN90ref'
dataName = 'Q95ref'
wnName = 'WaterNet0630'
if wnName == 'WaterNet0630':
    funcM = getattr(waterNetTest, wnName)
else:
    funcM = getattr(waterNet, wnName)
epWN = 250
epLSTM = 100
modelFile = '{}-{}-ep{}'.format(wnName, dataName, 200)
lstmOutName = '{}-{}'.format('QN90ref', trainSet)
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'

# train
DF = dbBasin.DataFrameBasin(dataName)

# waterNet
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

# data
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
# dataTup1 = DM1.getData()
# DM2 = dbBasin.DataModelBasin(
#     DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
# DM2.borrowStat(DM1)
# dataTup2 = DM2.getData()
DM = dbBasin.DataModelBasin(
    DF, subset='WYall', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM.borrowStat(DM1)
dataTup = DM.getData()

# model
nh = 16
ng = len(varXC)
ns = len(DF.siteNoLst)
nr = 5
model = funcM(nh, len(varXC), nr)
model = model.cuda()
torch.cuda.empty_cache()
torch.cuda.memory_reserved(0)
torch.cuda.memory_allocated(0)

# water net
model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))
model.eval()
[x, xc, y, yc] = dataTup

testBatch = 20
iS = np.arange(0, ns, testBatch)
iE = np.append(iS[1:], ns)
ns = y.shape[1]
t = DF.getT(testSet)
nt = len(t)
iS = np.arange(0, ns, testBatch)
iE = np.append(iS[1:], ns)
yP = np.ndarray([nt-nr+1, ns])
for k in range(len(iS)):
    print('batch {}'.format(k))
    xP = torch.from_numpy(x[:, iS[k]:iE[k], :]).float().cuda()
    xcP = torch.from_numpy(xc[iS[k]:iE[k]]).float().cuda()
    yOut = model(xP, xcP, outStep=False)
    temp = yOut.detach().cpu().numpy()
    yP[:, iS[k]:iE[k]] = temp[-nt+nr-1:, :]

# yOut, (QpR, QsR, QgR), (SfT, SsT, SgT) = model(xP, xcP, outStep=True)
# yP = yOut.detach().cpu().numpy()
# Qp = QpR.detach().cpu().numpy()
# Qs = QsR.detach().cpu().numpy()
# Qg = QgR.detach().cpu().numpy()
# Sf = SfT.detach().cpu().numpy()
# Ss = SsT.detach().cpu().numpy()
# Sg = SgT.detach().cpu().numpy()
