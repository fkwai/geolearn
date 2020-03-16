
import os
import time
import json
import numpy as np
import pandas as pd
import torch
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit

caseName = 'refBasins'
ratioTrain = 0.8
rho = 365
batchSize = 100
nEpoch = 100
hiddenSize = 64
modelFolder = os.path.join(kPath.dirWQ, 'modelA', caseName)
if not os.path.exists(modelFolder):
    os.mkdir(modelFolder)

# predict - point-by-point
modelFile = os.path.join(modelFolder, 'modelSeq_Ep' + str(nEpoch) + '.pt')
model = torch.load(modelFile)
nt = dictData['rho']
nd, ny = y.shape
batchSize = 1000
iS = np.arange(0, nd, batchSize)
iE = np.append(iS[1:], nd)
yOutLst = list()
xNorm = (xNorm - np.tile(statDict['xMean'], [nt, nd, 1])) / np.tile(
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
if not os.path.exists(targetFile):
    targetDf = pd.merge(dfOut, pd.DataFrame(data=y, columns=varC),
                        left_index=True, right_index=True)
    targetDf.to_csv(targetFile)
outFile = os.path.join(modelFolder, 'output_Ep' + str(nEpoch) + '.csv')
outDf = pd.merge(dfOut, pd.DataFrame(data=yOut, columns=varC),
                        left_index=True, right_index=True)
outDf.to_csv(outFile)