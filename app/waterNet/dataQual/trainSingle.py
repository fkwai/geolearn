
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from hydroDL.master import basinFull

dataName = 'G400'
DF = dbBasin.DataFrameBasin(dataName)

siteNoLst = DF.siteNoLst
codeLst = DF.varC
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM'],
                        siteNoLst=siteNoLst)
t1 = np.datetime64('1982-01-01')
t2 = np.datetime64('2010-01-01')
t3 = np.datetime64('2018-12-31')
indT1 = np.where(DF.t == t1)[0][0]
indT2 = np.where(DF.t == t2)[0][0]
indT3 = np.where(DF.t == t3)[0][0]

siteNo = '07241550'
code = '00915'
pLst = [100, 75, 50, 25]
testSet = '{}-A10'.format(siteNo)
indS = DF.siteNoLst.index(siteNo)
indC = DF.varC.index(code)
c = DF.c[:, indS, indC]
t = DF.t

doLst = ['test']
doLst = ['subset', 'train', 'test']


# subset
if 'subset' in doLst:
    temp = np.where(~np.isnan(c))[0]
    indT = temp[temp <= indT2]
    np.random.seed(0)
    np.random.shuffle(indT)
    for p in pLst:
        mask = np.ones([indT2-indT1+1, 1]).astype(bool)
        mask[indT[:int(len(indT)*p/100)], 0] = False
        subName = '{}-p{}'.format(siteNo, p)
        DF.saveSubset(subName, sd=str(t1), ed=str(t2),
                      siteNoLst=[siteNo], mask=mask)
    DF.saveSubset(testSet, sd=str(t2), ed=str(t3),
                  siteNoLst=[siteNo])


# train
if 'train' in doLst:
    label = 'QFPRT2C'
    for p in pLst:
        trainSet = '{}-p{}'.format(siteNo, p)
        varX = dbBasin.label2var(label.split('2')[0])
        mtdX = dbBasin.io.extractVarMtd(varX)
        varY = [code]
        mtdY = dbBasin.io.extractVarMtd([code])
        varXC = gageII.varLst
        mtdXC = dbBasin.io.extractVarMtd(varXC)
        varYC = None
        mtdYC = [code]
        outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
        dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                                     nEpoch=100, batchSize=[365, 10], nIterEp=50,
                                     varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                     mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
        basinFull.trainModel(outName)

# test
yPLst = list()
for p in pLst:
    trainSet = subName = '{}-p{}'.format(siteNo, p)
    outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
    yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet)
    yPLst.append(yP[:, 0, 0])

obs = DF.extractSubset(DF.c, testSet)[:, 0, indC]
tObs = t[indT2:indT3+1]
utils.stat.calCorr(yPLst[0], obs)
utils.stat.calCorr(yPLst[1], obs)
utils.stat.calCorr(yPLst[2], obs)
utils.stat.calCorr(yPLst[3], obs)


# plot
pLst = [25, 50, 75, 100]

fig, ax = plt.subplots(1, 1)
for yP, color in zip(yPLst, 'rmgb'):
    ax.plot(tObs, yP, color=color)
ax.plot(tObs, obs, 'k*')
ax.xaxis_date()
fig.show()

# plot
fig, axes = plt.subplots(len(pLst), 1)
for p, ax in zip(pLst, axes):
    subName = '{}-p{}'.format(siteNo, p)
    cc = DF.extractSubset(DF.c, subName)[:, 0, indC]
    print(np.sum(~np.isnan(cc)))
    tt = DF.t[indT1:indT2+1]
    axplot.plotTS(ax, tt, [cc], cLst=color)
fig.show()
