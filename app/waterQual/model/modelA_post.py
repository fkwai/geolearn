import os
import time
import json
import numpy as np
import pandas as pd
import torch
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.model import rnn, crit
from hydroDL.post import plot
import matplotlib.pyplot as plt
from datetime import datetime as dt
from random import randint

caseName = 'refBasins'
# caseName = 'temp'
nEpoch = 100
modelFolder = os.path.join(kPath.dirWQ, 'modelA', caseName)
dictData, info, x, y, c = waterQuality.loadData(caseName)

targetFile = os.path.join(modelFolder, 'target.csv')
dfT = pd.read_csv(targetFile, dtype={'siteNo': str})
outFile = os.path.join(modelFolder, 'output_Ep' + str(nEpoch) + '.csv')
dfP = pd.read_csv(outFile, dtype={'siteNo': str})

siteNoLst = dictData['siteNoLst']
varC = dictData['varC']
bb = True
while bb is True:
    iS = randint(0, len(siteNoLst))
    iC = randint(0, 20)
    # iS = 0
    # iC = 1
    t = dfT[dfT['siteNo'] == siteNoLst[iS]]['date']
    a = dfT[dfT['siteNo'] == siteNoLst[iS]][varC[iC]]
    b = dfP[dfP['siteNo'] == siteNoLst[iS]][varC[iC]]
    if not a.isna().all():
        fig, ax = plt.subplots(1, 1)
        fig, ax = plot.plotTS(t=t, y=[a, b], legLst=[
            varC[iC]+' obs', varC[iC]+' pred'])
        fig.show()
        bb = False
pass

# time series map
siteNoLst = dictData['siteNoLst']
varC = dictData['varC']
nP = len(siteNoLst)
nC = len(varC)
matRho1 = np.ndarray([nP, nC])
matRho2 = np.ndarray([nP, nC])
matRmse1 = np.ndarray([nP, nC])
matRmse2 = np.ndarray([nP, nC])
matN1 = np.ndarray([nP, nC])
matN2 = np.ndarray([nP, nC])

for iS, siteNo in enumerate(siteNoLst):
    print(iS)
    for iC, var in enumerate(varC):
        obs = dfT[dfT['siteNo'] == siteNoLst[iS]][varC[iC]].values
        pred = dfP[dfP['siteNo'] == siteNoLst[iS]][varC[iC]].values
        bTrain = dfT[dfT['siteNo'] == siteNoLst[iS]
                     ]['train'].values.astype(bool)
        # obs[bTrain==1].corr(pred[bTrain==1])
        # obs[bTrain==0].corr(pred[bTrain==0])
        ind1 = np.where(~np.isnan(obs) & bTrain)[0]
        ind2 = np.where(~np.isnan(obs) & ~bTrain)[0]
        matRho1[iS, iC] = np.corrcoef(obs[ind1], pred[ind1])[0, 1]
        matRho2[iS, iC] = np.corrcoef(obs[ind2], pred[ind2])[0, 1]
        matRmse1[iS, iC] = np.sqrt(np.mean((obs[ind1]-pred[ind1])**2))
        matRmse2[iS, iC] = np.sqrt(np.mean((obs[ind2]-pred[ind2])**2))
        matN1[iS, iC] = len(ind1)
        matN2[iS, iC] = len(ind2)
saveFile = os.path.join(modelFolder, 'statResult_Ep{}.npz'.format(nEpoch))
np.savez(saveFile, matRho1=matRho1, matRho2=matRho2, matRmse1=matRmse1, matRmse2=matRmse2, matN1=matN1, matN2=matN2)

# plot map