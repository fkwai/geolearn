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

dataName = 'weathering'
DF = dbBasin.DataFrameBasin(dataName)

yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(DF.t, yrIn)
t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
DF.createSubset('pkYr5', dateLst=t1)
DF.createSubset('rmYr5', dateLst=t2)


codeSel = ['00060']
varX = DF.varF
mtdX = dbBasin.io.extractVarMtd(varX)
varY = ['00060']
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

sd = '1982-01-01'
ed = '2009-12-31'
rho = 365
trainSet = 'rmYr5'
testSet = 'pkYr5'
outName = '{}-{}-t{}-{}'.format(dataName, 'testQ', rho, trainSet)
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             nEpoch=100, batchSize=[rho, 200], nIterEp=20,
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
# basinFull.trainModel(outName)

yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', reTest=True)

yO = DF.extractT(codeSel)

# load WRTDS
yW = np.ndarray(yO.shape)
for k, siteNo in enumerate(siteNoLst):
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
                            'QWRTDS-D', 'weathering-pkY5')
    saveFile = os.path.join(dirWRTDS, siteNo)
    df = pd.read_csv(saveFile, index_col=None).set_index('date')
    df.index = df.index.values.astype('datetime64[D]')
    indT = np.in1d(df.index.values, DF.t)
    yW[:, k, :] = df.loc[indT][codeSel].values

indT1, indT2, indS, mask = DF.readSubset(testSet)


d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeSel)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeSel)
for k in range(len(DF.siteNoLst)):
    dataPlot = [d1.Y[:, k, :], d2.Y[:, k, :], yW[:, k, :], yP[:, k, :]]
    cLst = ['grey', 'black', 'blue', 'red']
    fig, ax = plt.subplots(1, 1)
    axplot.plotTS(ax, DF.t, dataPlot,  cLst=cLst)
    fig.show()
