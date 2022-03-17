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

importlib.reload(axplot)
importlib.reload(figplot)

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

siteNo = '01184000'
siteNoLst = [siteNo]
dataName = 'weathering'
sd = '1982-01-01'
ed = '2018-12-31'
dataName = siteNo
freq = 'D'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)

yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(DF.t, yrIn)
t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
DF.createSubset('pkYr5', dateLst=t1)
DF.createSubset('rmYr5', dateLst=t2)


codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
label = 'test2'
varX = DF.varF
mtdX = dbBasin.io.extractVarMtd(varX)
varY = ['00060']
# mtdY = dbBasin.io.extractVarMtd(varY)
mtdY = ['QT']
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

sd = '1982-01-01'
ed = '2009-12-31'
rho = 365
trainSet = 'rmYr5'
testSet = 'pkYr5'
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

d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=varY)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=varY)
for k in range(len(DF.siteNoLst)):
    dataPlot = [yP[:, k, :], d1.Y[:, k, :], d2.Y[:, k, :]]
    cLst = ['red', 'grey', 'black']
    fig, axes = figplot.multiTS(
        DF.t, dataPlot, cLst=cLst)
    fig.show()
