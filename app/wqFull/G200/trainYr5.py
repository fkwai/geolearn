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
importlib.reload(utils)
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

# normalize
DF = dbBasin.DataFrameBasin('weathering')
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
DF = dbBasin.func.localNorm(DF, subset='rmD5')
DF.saveAs('weatheringNorm')
dataName = 'weatheringNorm'

label = 'test'
varX = DF.varF+DF.varQ
mtdX = dbBasin.io.extractVarMtd(varX)
varY = [c+'-N' for c in codeSel]
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst + [c+'-M' for c in codeSel] + [c+'-S' for c in codeSel]
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

sd = '1982-01-01'
ed = '2009-12-31'
rho = 365
trainSet = 'rmR20'
testSet = 'pkR20'
outName = '{}-{}-t{}-{}'.format(dataName, label, rho, trainSet)
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             nEpoch=100, batchSize=[rho, 200], nIterEp=20,
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
basinFull.trainModel(outName)

yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', reTest=True)

yO = DF.extractT(codeSel)


indT1, indT2, indS, mask = DF.readSubset(testSet)
mask = np.repeat(mask, len(codeSel), axis=2)

labelLst = list()
for ic, code in enumerate(codeSel):
    shortName = usgs.codePdf.loc[code]['shortName']
    temp = '{} {}'.format(
        code, shortName)
    labelLst.append(temp)

# deal with mean and std
yOut = np.ndarray(yP.shape)
for k, code in enumerate(codeSel):
    m = DF.g[:, DF.varG.index(code+'-M')]
    s = DF.g[:, DF.varG.index(code+'-S')]
    data = yP[:, :, k]
    yOut[:, :, k] = data*s+m


# # load WRTDS
# yW = np.ndarray(yO.shape)
# for k, siteNo in enumerate(siteNoLst):
#     dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
#                             'WRTDS-D', 'weathering-pkY5')
#     saveFile = os.path.join(dirWRTDS, siteNo)
#     df = pd.read_csv(saveFile, index_col=None).set_index('date')
#     df.index = df.index.values.astype('datetime64[D]')
#     indT = np.in1d(df.index.values, DF.t)
#     yW[:, k, :] = df.loc[indT][codeSel].values

d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeSel)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeSel)
for k in range(len(DF.siteNoLst)):
    dataPlot = [yOut[:, k, :], d1.Y[:, k, :], d2.Y[:, k, :]]
    cLst = ['red', 'grey', 'black']
    fig, axes = figplot.multiTS(
        DF.t, dataPlot, labelLst=labelLst, cLst=cLst)
    fig.show()

# correlation matrix
mat1 = np.ndarray([len(siteNoLst), len(codeSel), 4])
mat2 = np.ndarray([len(siteNoLst), len(codeSel), 4])
for indS, siteNo in enumerate(siteNoLst):
    for indC, code in enumerate(codeSel):
        stat = utils.stat.calStat(yOut[:, indS, indC], d2.Y[:, indS, indC])
        mat1[indS, indC, 0] = stat['Bias']
        mat1[indS, indC, 1] = stat['RMSE']
        mat1[indS, indC, 2] = stat['NSE']
        mat1[indS, indC, 3] = stat['Corr']

statStrLst = ['Bias', 'RMSE', 'NSE', 'Corr']
dataPlot = list()
for k, statStr in enumerate(statStrLst):
    temp = list()
    for ic, code in enumerate(codeSel):
        temp.append(mat1[:, ic, k])
    dataPlot.append(temp)
fig = figplot.boxPlot(dataPlot, widths=0.5, figsize=(12, 4),
                      label1=statStrLst, label2=codeSel, sharey=False)
fig.show()
