
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

dataName = 'G400Norm'
outName = dataName
trainSet = 'rmRT20'
testSet = 'pkRT20'

DF = dbBasin.DataFrameBasin(outName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=500)


# deal with mean and std
codeLst = usgs.newC
yOut = np.ndarray(yP.shape)
for k, code in enumerate(codeLst):
    m = DF.g[:, DF.varG.index(code+'-M')]
    s = DF.g[:, DF.varG.index(code+'-S')]
    data = yP[:, :, k]
    yOut[:, :, k] = data*s+m

# WRTDS
# yW = WRTDS.testWRTDS(dataName, trainSet, testSet, codeLst)
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npy')

# correlation matrix
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)
siteNoLst = DF.siteNoLst
mat1 = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
mat2 = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
for indS, siteNo in enumerate(siteNoLst):
    for indC, code in enumerate(codeLst):
        stat = utils.stat.calStat(yOut[:, indS, indC], d2.Y[:, indS, indC])
        stat2 = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
        mat1[indS, indC, 0] = stat['Bias']
        mat1[indS, indC, 1] = stat['RMSE']
        mat1[indS, indC, 2] = stat['NSE']
        mat1[indS, indC, 3] = stat['Corr']
        mat2[indS, indC, 0] = stat2['Bias']
        mat2[indS, indC, 1] = stat2['RMSE']
        mat2[indS, indC, 2] = stat2['NSE']
        mat2[indS, indC, 3] = stat2['Corr']

mat1[131, 11, :]

# selected sites
dictSiteName = 'dict{}.json'.format(dataName[:4])
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)


siteNo = dictSite['00915'][0]
indS = siteNoLst.index(siteNo)

code = '00915'
indC = codeLst.index(code)
fig, ax = plt.subplots(1, 1)
dataPlot = [yW[:, indS, indC], yOut[:, indS, indC],
            d1.Y[:, indS, indC], d2.Y[:, indS, indC]]
cLst = ['black', 'green', 'blue', 'red']
axplot.plotTS(ax, DF.t, dataPlot, cLst=cLst)
fig.show()


a = mat1[:, indC, 0]
b = mat1[:, indC, 1]
indS = np.where(a > 5)[0]
fig, ax = plt.subplots(1, 1)
ax.plot(a, b, '*')
fig.show()
mat1[131, 11, :]
