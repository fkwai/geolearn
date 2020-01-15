import os
import time
import json
import numpy as np
import pandas as pd
import torch
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit

caseName = 'temp'
ratioTrain = 0.8
rho = 365
batchSize = 100
nEpoch = 100
hiddenSize = 64
modelFolder = os.path.join(kPath.dirWQ, 'modelA', caseName)
if not os.path.exists(modelFolder):
    os.mkdir(modelFolder)

# load data
dictData, info, x, y, c = waterQuality.loadData(caseName)
# normalize and devide training/test
indTrain, indTest = waterQuality.divideTrain(info, 0.8)
statDict = dict(xMean=np.nanmean(x, axis=(0, 1)).tolist(),
                xStd=np.nanstd(x, axis=(0, 1)).tolist(),
                yMean=np.nanmean(y, axis=0).tolist(),
                yStd=np.nanstd(y, axis=0).tolist(),
                cMean=np.nanmean(c, axis=0).tolist(),
                cStd=np.nanstd(c, axis=0).tolist())
for s in ['xStd', 'yStd', 'cStd']:
    statDict[s] = [temp if temp > 1e-5 else 1 for temp in statDict[s]]

with open(os.path.join(modelFolder, 'stat.json'), 'w') as fp:
    json.dump(statDict, fp)

xTrain = x[:, indTrain, :]
yTrain = y[indTrain, :]
cTrain = c[indTrain, :]
nt, nd, nx = xTrain.shape
nd, ny = yTrain.shape
nd, nc = cTrain.shape
xNorm = (xTrain - np.tile(statDict['xMean'], [nt, nd, 1])) / np.tile(
    statDict['xStd'], [nt, nd, 1])
yNorm = (yTrain - np.tile(statDict['yMean'], [nd, 1])) / np.tile(
    statDict['yStd'], [nd, 1])
cNorm = (cTrain - np.tile(statDict['cMean'], [nd, 1])) / np.tile(
    statDict['cStd'], [nd, 1])


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
model.zero_grad()
model.train()
# time.sleep(5)
for iEp in range(1, nEpoch + 1):
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
            print('iteration Failed')
    lossEp = lossEp / nIterEp
    ct = time.time() - t0
    logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(iEp, lossEp, ct)
    print(logStr)
    lossEpLst.append(lossEp)
modelFile = os.path.join(modelFolder, 'modelSeq_Ep' + str(nEpoch) + '.pt')
# torch.save(model, modelFile)

# predict - point-by-point
# modelFile = os.path.join(modelFolder, 'modelSeq_Ep' + str(nEpoch) + '.pt')
# model = torch.load(modelFile)
nt = dictData['rho']
nd, ny = y.shape
batchSize = 1000
iS = np.arange(0, nd, batchSize)
iE = np.append(iS[1:], nd)
yOutLst = list()
xNorm = (x - np.tile(statDict['xMean'], [nt, nd, 1])) / np.tile(
    statDict['xStd'], [nt, nd, 1])
cNorm = (c - np.tile(statDict['cMean'], [nd, 1])) / np.tile(
    statDict['cStd'], [nd, 1])

for k in range(len(iS)):
    print('batch: '+str(k))
    xT = torch.from_numpy(np.concatenate(
        [xNorm[:, iS[k]:iE[k], :], np.tile(cNorm[iS[k]:iE[k], :], [nt, 1, 1])], axis=-1)).float()
    if torch.cuda.is_available():
        xT = xT.cuda()
        model = model.cuda()
    yT = model(xT)[-1, :, :]
    yOutLst.append(yT.detach().cpu().numpy())
yOut = np.concatenate(yOutLst, axis=0)
yOut = yOut * np.tile(statDict['yStd'], [nd, 1]) +\
    np.tile(statDict['yMean'], [nd, 1])

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
outFile = os.path.join(modelFolder, 'output_Ep' + str(nEpoch) + '.csv')
outDf = pd.merge(dfOut, pd.DataFrame(data=yOut, columns=varC),
                 left_index=True, right_index=True)
outDf.to_csv(outFile)
