
import os
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII, gridMET
from hydroDL.master import basinFull
import numpy as np
from hydroDL import utils
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from hydroDL.model import waterNetGlobal
import importlib

importlib.reload(waterNetGlobal)
importlib.reload(crit)
saveDir = r'C:\Users\geofk\work\temp\waternet'

dataName = 'HBN_Q90ref'
# dataName = 'temp'
DF = dbBasin.DataFrameBasin(dataName)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'LAI']
mtdX = ['skip' for k in range(4)]+['norm']
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
# mtdXC = dbBasin.io.extractVarMtd(varXC)
# mtdXC = ['QT' for var in varXC]
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
DM3 = dbBasin.DataModelBasin(
    DF, subset='all', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM3.borrowStat(DM1)
dataTup3 = DM3.getData()

# model
nh = 16
ng = len(varXC)
ns = len(DF.siteNoLst)
model = waterNetGlobal.WaterNet3(nh, 1, ng)
model = model.cuda()
# optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters())
# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.LogLoss2D().cuda()

outDir = os.path.join(saveDir, 'HBN36', 'wn2-ep1000-save')
modelFile = os.path.join(outDir, 'model')
model.load_state_dict(torch.load(modelFile))

model.eval()
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
t = DF.getT(testSet)
# yP = model(xP, xcP).detach().cpu().numpy()
yOut, (q1Out, q2Out, q3Out) = model(xP, xcP, outQ=True)
w = model.fc(xcP)
gaOut = torch.softmax(w[:, nh*6:nh*7], dim=1)

yP = yOut.detach().cpu().numpy()
q1P = q1Out.detach().cpu().numpy()
q2P = q2Out.detach().cpu().numpy()
q3P = q3Out.detach().cpu().numpy()
ga = gaOut.detach().cpu().numpy()
model.zero_grad()

w = model.fc(xcP)
gm = torch.exp(w[:, :nh])+1
ge = torch.sigmoid(w[:, nh:nh*2])*2
k2 = torch.sigmoid(w[:, nh*2:nh*3])
k23 = torch.sigmoid(w[:, nh*3:nh*4])
k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
gl = torch.exp(w[:, nh*5:nh*6])*2
ga = torch.softmax(w[:, nh*6:nh*7], dim=1)
qb = torch.relu(w[:, 7:8])

a = ga.detach().cpu().numpy()
x = k3.detach().cpu().numpy()
b = qb.detach().cpu().numpy()
d = np.sum(x/b*a, axis=1)
for var in DF.varG:
    y = DF.g[:, DF.varG.index(var)]
    print(np.corrcoef(d, y)[0, 1], var)

y = DF.g[:, DF.varG.index('PERMAVE')]
fig, ax = plt.subplots(1, 1)
ax.plot(d, y, '*')
fig.show()
np.corrcoef(d, y)

DF.varG
