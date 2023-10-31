import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.master import basinFull
from hydroDL.master import slurm
import time

codeLst = usgs.varC

trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'


epLst = range(20, 501, 20)
errLst = list()
corrLst1, corrLst2, nashLst1, nashLst2 = [list() for x in range(4)]

# LSTM
label = 'QFT2C'
t0 = time.time()
dataName = 'rmTK-{}'.format('B200')
DF = dbBasin.DataFrameBasin(dataName)
outName = '{}-{}-{}'.format(dataName, label, trainSet)
dictMaster = basinFull.loadMaster(outName)
outFolder = basinFull.nameFolder(outName)
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
tabOut1 = pd.DataFrame(index=DF.siteNoLst, columns=epLst)
tabOut2 = pd.DataFrame(index=DF.siteNoLst, columns=epLst)
for ep in epLst:
    if not os.path.exists(os.path.join(outFolder, 'modelState_ep{}'.format(ep))):
        # if not os.path.exists(os.path.join(outFolder, 'testP-{}-Ep{}.npz'.format(trainSet,ep))):
        errLst.append(outName)
    else:
        yP1, ycP1 = basinFull.testModel(
            outName, testSet=trainSet, ep=ep, DF=DF, batchSize=20
        )
        yP2, ycP2 = basinFull.testModel(
            outName, testSet=testSet, ep=ep, DF=DF, batchSize=20
        )
        corr1 = utils.stat.calCorr(yP1, obs1)
        corr2 = utils.stat.calCorr(yP2, obs2)
        nash1 = utils.stat.calNash(yP1, obs1)
        nash2 = utils.stat.calNash(yP2, obs2)
        corrLst1.append(corr1)
        corrLst2.append(corr2)
        nashLst1.append(nash1)
        nashLst2.append(nash2)
        print('{} ep{} {:.2f} '.format(label, ep, time.time() - t0))


# load WRTDS
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('rmTK-B200', trainSet, 'all.npz')
yW = np.load(os.path.join(dirWRTDS, fileName))['yW']
yW1 = DF.extractSubset(yW, trainSet)
yW2 = DF.extractSubset(yW, testSet)
nashW1 = utils.stat.calNash(yW1, obs1)
nashW2 = utils.stat.calNash(yW2, obs2)
corrW1 = utils.stat.calCorr(yW1, obs1)
corrW2 = utils.stat.calCorr(yW2, obs2)



for code in codeLst:
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    iC = codeLst.index(code)
    dataPlot1, dataPlot2 = list(), list()
    # for corr1,corr2 in zip(corrLst1,corrLst2):
    for corr1, corr2 in zip(nashLst1, nashLst2):
        temp1 = corr1[:, iC]
        dataPlot1.append(temp1[~np.isnan(temp1)])
        temp2 = corr2[:, iC]
        dataPlot2.append(temp2[~np.isnan(temp2)])
    temp1=nashW1[:,iC]
    dataPlot1.append(temp1[~np.isnan(temp1)])
    temp2=nashW2[:,iC]
    dataPlot2.append(temp2[~np.isnan(temp2)])
    bp1 = axes[0].boxplot(dataPlot1, showfliers=False)
    bp2 = axes[1].boxplot(dataPlot2, showfliers=False)
    fig.suptitle('{} {}'.format(code, usgs.codePdf.loc[code]['shortName']))
    fig.show()
