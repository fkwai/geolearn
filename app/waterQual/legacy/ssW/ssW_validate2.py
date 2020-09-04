from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.model import rnn, crit, trainTS
import torch

dataName = 'ssW'
wqData = waterQuality.DataModelWQ(dataName)

code = '00945'
label = 'plain'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)

outFolder = basins.nameFolder(outName)
dictP = basins.loadMaster(outName)

# load data
rmFlag = dictP['rmFlag'] if 'rmFlag' in dictP else False
wqData = waterQuality.DataModelWQ(dictP['dataName'], rmFlag)
varTup = (dictP['varX'], dictP['varXC'], dictP['varY'], dictP['varYC'])
dataTup, statTup = wqData.transIn(
    subset=dictP['trainName'], varTup=varTup)
dataTup = trainTS.dealNaN(dataTup, dictP['optNaN'])
# wrapStat(outName, statTup)
[nx, nxc, ny, nyc, nt, ns] = trainTS.getSize(dataTup)
model = basins.loadModel(outName, ep=500)

lossFun = crit.RmseLoss()
lossFun = lossFun.cuda()
model = model.cuda()

# training parts
dataLst = dataTup
sizeLst = trainTS.getSize(dataLst)
[nx, nxc, ny, nyc, nt, ns] = sizeLst
rho, nbatch = dictP['batchSize']
rho = nt
batchSize = [rho, nbatch]
xT, yT = trainTS.subsetRandom(dataLst, batchSize, sizeLst)
yP = model(xT)
loss = lossFun(yP, yT)
print(loss)

# testing parts
x, xc, y, yc = dataTup
iS = np.arange(0, ns, 2000)
iE = np.append(iS[1:], ns)
k = 0
xT = torch.from_numpy(np.concatenate(
    [x[:, iS[k]:iE[k], :], np.tile(xc[iS[k]:iE[k], :], [nt, 1, 1])], axis=-1)).float()
ycTemp = np.full([rho, 2000, nyc], np.nan)
ycTemp[-1, :, :] = yc[iS[k]:iE[k], :] if yc is not None else np.nan
yT = torch.from_numpy(np.concatenate(
    [y[:, iS[k]:iE[k], :], ycTemp], axis=-1)).float()
xT = xT.cuda()
yT = yT.cuda()
yP = model(xT)
loss = lossFun(yP, yT)
print(loss)
