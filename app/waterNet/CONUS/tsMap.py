
import matplotlib.gridspec as gridspec
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from hydroDL import utils
import os
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
import pandas as pd
from hydroDL.model import waterNetGlobal
from hydroDL.master import basinFull
import importlib

importlib.reload(waterNetGlobal)
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
model = waterNetGlobal.WaterNet3(nh, 1, ng)
model = model.cuda()
sn = 1e-8

# water net
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'model-{}-ep{}'.format('QN90ref', 500)
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
for k in range(len(iS)):
    print('batch {}'.format(k))
    yOut = model(xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]])
    yP[:, iS[k]:iE[k]] = yOut.detach().cpu().numpy()
model.zero_grad()


# LSTM
outName = '{}-{}'.format('QN90ref', trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=True, ep=1000)
yL = yL[:, :, 0]

nash1 = utils.stat.calNash(yP, y[:, :, 0])
corr1 = utils.stat.calCorr(yP, y[:, :, 0])
nash2 = utils.stat.calNash(yL, y[:, :, 0])
corr2 = utils.stat.calCorr(yL, y[:, :, 0])

lat, lon = DF.getGeo()

importlib.reload(mapplot)
importlib.reload(axplot)


def funcM():
    figM = plt.figure()
    gsM = gridspec.GridSpec(3, 1)
    axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash1)
    axM1 = mapplot.mapPoint(figM, gsM[1, 0], lat, lon, nash2)
    axM2 = mapplot.mapPoint(figM, gsM[2, 0], lat, lon, nash2-nash1)
    axM = np.array([axM0, axM1, axM2])
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = DF.siteNoLst[iP]
    t = DF.getT(testSet)
    legLst = ['waterNet {:.2f} {:.2f}'.format(nash1[iP], corr1[iP]),
              'LSTM {:.2f} {:.2f}'.format(nash2[iP], corr2[iP]),
              'obs']
    axplot.plotTS(axP, t, [y[:, iP, 0], yP[:, iP], yL[:, iP]],
                  lineW=[2, 1, 1], cLst='krb', legLst=legLst)
    strTitle = ('{}'.format(DF.siteNoLst[iP]))
    axP.set_title(strTitle)


figM, figP = figplot.clickMap(funcM, funcP)

figM, axM, figP, axP, lon, lat = funcM()
