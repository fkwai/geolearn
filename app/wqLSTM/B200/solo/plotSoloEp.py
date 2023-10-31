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
labelLst = ['QFT2C', 'QT2C', 'FT2QC']
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'

epLst = range(20, 501, 20)
for code in codeLst:
    try:
        label='QFT2C'
        trainSet = 'rmYr5b0'
        testSet = 'pkYr5b0'
        dataName = '{}-{}'.format(code, 'B200')
        outName = '{}-{}-{}'.format(dataName, label, trainSet)
        outFolder = basinFull.nameFolder(outName)
        tabOut1 = pd.read_csv(os.path.join(outFolder, 'corrEpTrain.csv'), index_col=0)
        tabOut2 = pd.read_csv(os.path.join(outFolder, 'corrEpTest.csv'), index_col=0)
        # boxplot
        fig, axes = plt.subplots(2, 1)
        axes[0].boxplot(tabOut1.values,showfliers=False)
        axes[1].boxplot(tabOut2.values,showfliers=False)
        fig.suptitle('{} {}'.format(code, label))
        fig.show()


for label in labelLst:
    for code in codeLst:
        t0 = time.time()
        dataName = '{}-{}'.format(code, 'B200')
        DF = dbBasin.DataFrameBasin('{}-{}'.format(code, 'B200'))
        outName = '{}-{}-{}'.format(dataName, label, trainSet)
        dictMaster = basinFull.loadMaster(outName)
        outFolder = basinFull.nameFolder(outName)
        matObs = DF.extractT([code])
        obs1 = DF.extractSubset(matObs, trainSet)
        obs2 = DF.extractSubset(matObs, testSet)
        tabOut1 = pd.DataFrame(index=DF.siteNoLst, columns=epLst)
        tabOut2 = pd.DataFrame(index=DF.siteNoLst, columns=epLst)
        for ep in epLst:
            yP1, ycP1 = basinFull.testModel(
                outName, testSet=trainSet, ep=ep, DF=DF, batchSize=20
            )
            yP2, ycP2 = basinFull.testModel(
                outName, testSet=testSet, ep=ep, DF=DF, batchSize=20
            )
            corr1 = utils.stat.calCorr(yP1, obs1)
            corr2 = utils.stat.calCorr(yP2, obs2)
            tabOut1[ep] = corr1
            tabOut2[ep] = corr2
            print('{} {} ep{} {:.2f} '.format(code, label, ep, time.time() - t0))
        tabOut1.to_csv(os.path.join(outFolder, 'corrEpTrain.csv'))
        tabOut2.to_csv(os.path.join(outFolder, 'corrEpTest.csv'))


# errLst1 = list()
# errLst2 = list()
# errLst3 = list()
# test for error
# epLst= range(20,501,20)
# for label in labelLst:
#     for code in codeLst:
#         for ep in epLst:
#             dataName = '{}-{}'.format(code, 'B200')
#             outName = '{}-{}-{}'.format(dataName, label, trainSet)
#             dictMaster = basinFull.loadMaster(outName)
#             outFolder = basinFull.nameFolder(outName)
#             if not os.path.exists(os.path.join(outFolder, 'modelState_ep{}'.format(ep))):
#                 errLst1.append(outName)
#             else:
#                 if not os.path.exists(
#                     os.path.join(outFolder, 'modelState_ep{}'.format(ep))
#                 ):
#                     errLst2.append(outName)
#                 else:
#                     outLst.append(outName)
#             if not os.path.exists(os.path.join(outFolder, 'master.json')):
#                 errLst3.append(outName)
