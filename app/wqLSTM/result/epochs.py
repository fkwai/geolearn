
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS

dataName = 'G200N'
label = 'QFPRT2C'
trainSet = 'rmL20'
testSet = 'pkL20'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
codeLst = usgs.newC
siteNoLst = DF.siteNoLst
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)

# selected sites
dictSiteName = 'dict{}.json'.format(dataName[:4])
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)

epLst = list(range(100, 2100, 100))
corrMat = np.full([len(siteNoLst), len(codeLst), len(epLst)], np.nan)
for iEp, ep in enumerate(epLst):
    yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=ep)
    # deal with mean and std
    codeLst = usgs.newC
    yOut = np.ndarray(yP.shape)
    for indC, code in enumerate(codeLst):
        m = DF.g[:, DF.varG.index(code+'-M')]
        s = DF.g[:, DF.varG.index(code+'-S')]
        yOut[:, :, indC] = yP[:, :, indC]*s+m
    for indC, code in enumerate(codeLst):
        indS = [siteNoLst.index(siteNo)
                for siteNo in dictSite[code] if siteNo in siteNoLst]
        corr = utils.stat.calCorr(yOut[:, indS, indC], d2.Y[:, indS, indC])
        corrMat[indS, indC, iEp] = corr

dataPlot = np.nanmean(corrMat, axis=0)
fig, ax = plt.subplots(1, 1)
for ic, code in enumerate(codeLst):
    label = code + ' ' + usgs.codePdf.loc[code]['shortName']
    ax.plot(epLst, np.nanmean(corrMat[:, ic, :], axis=0), label=label)
ax.legend(bbox_to_anchor=(1.1, 1.05))
fig.show()
