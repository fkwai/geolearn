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
from hydroDL.model import waterNetTest
from hydroDL.master import basinFull
import importlib

importlib.reload(waterNetTest)
importlib.reload(crit)

dataName = 'QN90ref'
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

# model
nh = 16
ng = len(varXC)
ns = len(DF.siteNoLst)
model = waterNetTest.WaterNet(nh, 1, ng)
model = model.cuda()
sn = 1e-8

# water net
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'wn1104-{}-ep{}'.format('QN90ref', 500)
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
yP = np.ndarray([nt, ns])
q1P = np.ndarray([nt, ns, nh])
q2P = np.ndarray([nt, ns, nh])
q3P = np.ndarray([nt, ns, nh])
for k in range(len(iS)):
    print('batch {}'.format(k))
    yOut, (q1Out, q2Out, q3Out) = model(
        xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]], outQ=True)
    yP[:, iS[k]:iE[k]] = yOut.detach().cpu().numpy()
    q1P[:, iS[k]:iE[k], :] = q1Out.detach().cpu().numpy()
    q2P[:, iS[k]:iE[k], :] = q2Out.detach().cpu().numpy()
    q3P[:, iS[k]:iE[k], :] = q3Out.detach().cpu().numpy()
model.zero_grad()

# load parameters
x = xP
xc = xcP
P, E, T1, T2, LAI = [x[:, :, k] for k in range(5)]
w = model.fc(xc)
xcT = torch.cat([LAI[:, :, None], torch.tile(xc, [nt, 1, 1])], dim=-1)
v = model.fcT(xcT)
gm = torch.exp(w[:, :nh])+1
k1 = torch.sigmoid(w[:, nh:nh*2])
k2 = torch.sigmoid(w[:, nh*2:nh*3])
k23 = torch.sigmoid(w[:, nh*3:nh*4])
k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
gl = torch.exp(w[:, nh*5:nh*6])*2
ga = torch.softmax(w[:, nh*6:nh*7], dim=1)
qb = torch.relu(w[:, nh*7:nh*8])
ge = torch.sigmoid(w[:, nh*8:nh*9])*5
ve = torch.sigmoid(w[:, nh*9:nh*10])*5
vi = torch.relu(v[:, :, :nh])
vk = torch.sigmoid(v[:, :, nh:nh*2])


siteNo = '01491000'
indS = DF.siteNoLst.index(siteNo)
q1 = q1P[:, indS, :]
q2 = q2P[:, indS, :]
q3 = q3P[:, indS, :]
r1 = k1[indS, :].detach().cpu().numpy()
r2 = k2[indS, :].detach().cpu().numpy()
r3 = k3[indS, :].detach().cpu().numpy()
a = ga[indS, :].detach().cpu().numpy()

fig, axes = plt.subplots(3, 1)
t = DF.getT(testSet)
axes[0].plot(t, q1)
axes[1].plot(t, q2)
axes[2].plot(t, q3)
fig.show()

te = -1
c1 = 1/r1
c2 = 1/r2
c3 = 1/r3
c1[c1 > te] = 0
c2[c2 > te] = 0
c3[c3 > te] = 10

# c1 = np.random.rand(nh)
# c2 = np.random.rand(nh)
# c3 = np.random.rand(nh)

q = np.sum(q1+q2+q3, axis=1)
c = np.sum(q1*c1+q2*c2+q3*c3, axis=1)/np.sum(q1+q2+q3, axis=1)
fig, ax = plt.subplots(1, 1)
ax.plot(np.log(q), c, '*')
fig.show()


mat = np.stack([a, r1, r2, r3, c1, c2, c3])
fig = plt.figure()
# indLst = [[0], [1, 2, 3], [4, 5, 6]]
gs = gridspec.GridSpec(7, 1)
ax = fig.add_subplot(gs[0, 0])
axplot.plotHeatMap(ax, mat[0:1, :]*100, labLst=[['area'], []], fmt='{:.0f}')
ax.set_xticks([])
ax = fig.add_subplot(gs[1:4, 0])
axplot.plotHeatMap(ax, mat[1:4, :], labLst=[
                   ['k1', 'k2', 'k3'], []], fmt='{:.2f}')
ax.set_xticks([])
ax = fig.add_subplot(gs[4:7, 0])
axplot.plotHeatMap(ax, mat[4:7, :], labLst=[
                   ['r1', 'r2', 'r3'], []], fmt='{:.0f}')
ax.set_xticks([])
fig.show()


'{:.0E}'.format(0.003)
