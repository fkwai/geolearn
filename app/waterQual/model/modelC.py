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
batchSize = 100
nEpoch = 500
saveEpoch = 100
resumeEpoch = 0
hiddenSize = 256
modelFolder = os.path.join(kPath.dirWQ, 'modelA', caseName)
if not os.path.exists(modelFolder):
    os.mkdir(modelFolder)

# load data
dictData, info, x, y, c = waterQuality.loadData(caseName)
mtdLstX = ['log-norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm']
mtdLstY = list(usgs.dictStat.values())
mtdLstC = list(gageII.dictStat.values())
x = x[:, :, 1:]

# normalize
xNorm = np.ndarray(x.shape)
yNorm = np.ndarray(y.shape)
cNorm = np.ndarray(c.shape)
statLstX = list()
statLstY = list()
statLstC = list()
for k, mtd in enumerate(mtdLstX):
    xNorm[:, :, k], stat = transform.transIn(x[:, :, k], mtd)
    statLstX.append(stat)
for k, mtd in enumerate(mtdLstY):
    yNorm[:, k], stat = transform.transIn(y[:, k], mtd)
    statLstY.append(stat)
for k, mtd in enumerate(mtdLstC):
    cNorm[:, k], stat = transform.transIn(c[:, k], mtd)
    statLstC.append(stat)
dictStat = dict(mtdLstX=mtdLstX, statLstX=statLstX, mtdLstY=mtdLstY,
                statLstY=statLstY, mtdLstC=mtdLstC, statLstC=statLstC)
with open(os.path.join(modelFolder, 'stat.json'), 'w') as fp:
    json.dump(dictStat, fp)

# devide training/test
indTrain, indTest = waterQuality.divideTrain(info, 0.8)
xTrain = xNorm[:, indTrain, :]
yTrain = yNorm[indTrain, :]
cTrain = cNorm[indTrain, :]
nt, nd, nx = xTrain.shape
nd, ny = yTrain.shape
nd, nc = cTrain.shape


# random subset
def subset(x, y, c):
    iR = np.random.randint(0, nd, [batchSize])
    yTensor = torch.from_numpy(y[None, iR, :]).float()
    cTemp = np.tile(c[iR, :], [nt, 1, 1])
    xTemp = x[nt-rho:rho, iR, :]
    xTensor = torch.from_numpy(np.concatenate([xTemp, cTemp], axis=-1)).float()
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
    model = rnn.CudnnLstmModel(nx=nx+nc, ny=ny, hiddenSize=hiddenSize)
lossFun = crit.RmseEnd()
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
        xT, yT = subset(xNorm, yNorm, cNorm)
        try:
            yP = model(xT)
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
        modelFile = os.path.join(modelFolder, 'model_Ep' + str(iEp) + '.pt')
        torch.save(model, modelFile)
        model.eval()

        # predict - point-by-point
        # modelFile = os.path.join(modelFolder, 'model_Ep' + str(nEpoch) + '.pt')
        nt = dictData['rho']
        nd, ny = y.shape
        iS = np.arange(0, nd, batchSize)
        iE = np.append(iS[1:], nd)
        yOutLst = list()
        for k in range(len(iS)):
            print('batch: '+str(k))
            xT = torch.from_numpy(np.concatenate(
                [xNorm[:, iS[k]:iE[k], :], np.tile(cNorm[iS[k]:iE[k], :], [nt, 1, 1])], axis=-1)).float()
            if torch.cuda.is_available():
                xT = xT.cuda()
                modelTest = modelTest.cuda()
            yT = modelTest(xT)[-1, :, :]
            yOutLst.append(yT.detach().cpu().numpy())
        temp = np.concatenate(yOutLst, axis=0)
        yOut = np.ndarray(temp.shape)
        for k in range(ny):
            yOut[:, k] = transform.transOut(
                temp[:, k], mtdLstY[k], statLstY[k])

        # save output
        dfOut = info
        dfOut['train'] = np.nan
        dfOut['train'][indTrain] = 1
        dfOut['train'][indTest] = 0
        varC = dictData['varC']
        targetFile = os.path.join(modelFolder, 'target.csv')
        # if not os.path.exists(targetFile):
        targetDf = pd.merge(dfOut, pd.DataFrame(data=y, columns=varC),
                            left_index=True, right_index=True)
        targetDf.to_csv(targetFile)
        outFile = os.path.join(modelFolder, 'output_Ep' + str(iEp) + '.csv')
        outDf = pd.merge(dfOut, pd.DataFrame(data=yOut, columns=varC),
                         left_index=True, right_index=True)
        outDf.to_csv(outFile)
        model.train()
