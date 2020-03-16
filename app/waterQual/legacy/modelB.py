import os
import time
import json
import numpy as np
import pandas as pd
import torch
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit
from hydroDL.data import usgs, gageII, transform

caseName = 'refBasins'
# caseName = 'temp'

ratioTrain = 0.8
rho = 365
batchSize = 1000
nEpoch = 500
saveEpoch = 100
resumeEpoch = 0
hiddenSize = 256
modelFolder = os.path.join(kPath.dirWQ, 'modelB', caseName)
if not os.path.exists(modelFolder):
    os.mkdir(modelFolder)

# load data
dictData, info, x, y, c = waterQuality.loadData(caseName)

# add streamflow into y - temp code
xc = c
yc = y
y = x[:, :, 0:1]
x = x[:, :, 1:]

mtdLstX = ['log-norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm']
mtdLstXC = list(gageII.dictStat.values())
mtdLstY = ['log-norm']
mtdLstYC = list(usgs.dictStat.values())


# normalize
xNorm = np.ndarray(x.shape)
xcNorm = np.ndarray(xc.shape)
yNorm = np.full(y.shape, np.nan)
ycNorm = np.full(yc.shape, np.nan)
statLstXC = list()
statLstX = list()
statLstYC = list()
statLstY = list()
for k, mtd in enumerate(mtdLstX):
    xNorm[:, :, k], stat = transform.transIn(x[:, :, k], mtd)
    statLstX.append(stat)
for k, mtd in enumerate(mtdLstY):
    yNorm[:, :, k], stat = transform.transIn(y[:, :, k], mtd)
    statLstY.append(stat)
for k, mtd in enumerate(mtdLstXC):
    xcNorm[:, k], stat = transform.transIn(xc[:, k], mtd)
    statLstXC.append(stat)
for k, mtd in enumerate(mtdLstYC):
    ycNorm[:, k], stat = transform.transIn(yc[:, k], mtd)
    statLstYC.append(stat)

dictStat = dict(mtdLstX=mtdLstX, statLstX=statLstX, mtdLstXC=mtdLstXC, statLstXC=statLstXC,
                mtdLstY=mtdLstY, statLstY=statLstY, mtdLstYC=mtdLstYC, statLstYC=statLstYC)
with open(os.path.join(modelFolder, 'stat.json'), 'w') as fp:
    json.dump(dictStat, fp)

# devide training/test
indTrain, indTest = waterQuality.divideTrain(info, 0.8)
xTrain = xNorm[:, indTrain, :]
yTrain = yNorm[:, indTrain, :]
xcTrain = xcNorm[indTrain, :]
ycTrain = ycNorm[indTrain, :]
nt, nd, nx = xTrain.shape
nt, nd, ny = yTrain.shape
nd, nxc = xcTrain.shape
nd, nyc = ycTrain.shape


# random subset


def subset(x, xc, y, yc):
    iR = np.random.randint(0, nd, [batchSize])
    xTemp = x[nt-rho:rho, iR, :]
    xcTemp = np.tile(xc[iR, :], [nt, 1, 1])
    xTensor = torch.from_numpy(np.concatenate(
        [xTemp, xcTemp], axis=-1)).float()
    yTemp = y[nt-rho:rho, iR, :]
    ycTemp = np.full([rho, batchSize, nyc], np.nan)
    ycTemp[-1, :, :] = yc[iR, :]
    yTensor = torch.from_numpy(np.concatenate(
        [yTemp, ycTemp], axis=-1)).float()
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


# model
if resumeEpoch != 0:
    modelFile = os.path.join(
        modelFolder, 'model_Ep' + str(resumeEpoch) + '.pt')
    model = torch.load(modelFile)
else:
    model = rnn.CudnnLstmModel(nx=nx+nxc, ny=ny+nyc, hiddenSize=hiddenSize)
# lossFun = crit.RmseMix()
lossFun = crit.RmseLoss()
if torch.cuda.is_available():
    lossFun = lossFun.cuda()
    model = model.cuda()
optim = torch.optim.Adadelta(model.parameters())

# training
if batchSize > nd:
    nIterEp = 1
else:
    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize / nd)))
lossEp = 0
lossEpLst = list()
t0 = time.time()
model.train()
model.zero_grad()
# time.sleep(5)
for iEp in range(resumeEpoch+1, nEpoch + 1):
    lossEp = 0
    t0 = time.time()
    for iIter in range(nIterEp):
        xT, yT = subset(xNorm, xcNorm, yNorm, ycNorm)
        try:
            yP = model(xT)
            # loss = lossFun(yP[:, :, 0:1], yP[-1, :, 1:],
            #                yT[:, :, 0:1], yT[-1, :, 1:])
            loss = lossFun(yP, yT)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
        except:
            print('iteration Failed: iter {} ep {}'.format(iIter, iEp))
    lossEp = lossEp / nIterEp
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, lossEp, ct)
    print(logStr)
    lossEpLst.append(lossEp)

    if iEp % saveEpoch == 0:
        model.eval()
        modelFile = os.path.join(modelFolder, 'model_Ep' + str(iEp) + '.pt')
        torch.save(model, modelFile)

        # predict - point-by-point
        # modelFile = os.path.join(modelFolder, 'model_Ep' + str(nEpoch) + '.pt')
        # model = torch.load(modelFile)
        nt = dictData['rho']
        nd, nyc = yc.shape
        iS = np.arange(0, nd, batchSize)
        iE = np.append(iS[1:], nd)
        yPLst = list()
        ycPLst = list()
        for k in range(len(iS)):
            print('batch: '+str(k))
            xT = torch.from_numpy(np.concatenate(
                [xNorm[:, iS[k]:iE[k], :], np.tile(xcNorm[iS[k]:iE[k], :], [nt, 1, 1])], axis=-1)).float()
            if torch.cuda.is_available():
                xT = xT.cuda()
                model = model.cuda()
            # yT = model(xT)[:, :, 0]
            ycT = model(xT)[-1, :, 1:]
            # yPLst.append(yT.detach().cpu().numpy())
            ycPLst.append(ycT.detach().cpu().numpy())
        # yP = np.concatenate(yPLst, axis=1)
        ycP = np.concatenate(ycPLst, axis=0)
        # yOut = np.ndarray(yP.shape)
        ycOut = np.ndarray(ycP.shape)
        # yOut = transform.transOut(yP, mtdLstY[0], statLstY[0])
        for k in range(nyc):
            ycOut[:, k] = transform.transOut(
                ycP[:, k], mtdLstYC[k], statLstYC[k])

        # save output
        print(2)
        dfOut = info
        dfOut['train'] = np.nan
        dfOut['train'][indTrain] = 1
        dfOut['train'][indTest] = 0
        varC = dictData['varC']
        targetFile = os.path.join(modelFolder, 'target.csv')
        # if not os.path.exists(targetFile):
        targetDf = pd.merge(dfOut, pd.DataFrame(data=yc, columns=varC),
                            left_index=True, right_index=True)
        targetDf.to_csv(targetFile)
        outFile = os.path.join(modelFolder, 'output_Ep' + str(iEp) + '.csv')
        outDf = pd.merge(dfOut, pd.DataFrame(data=ycOut, columns=varC),
                         left_index=True, right_index=True)
        outDf.to_csv(outFile)
        model.train()
