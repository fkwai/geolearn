
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

siteNo = '01434025'
code = '00915'
indS = DF.siteNoLst.index(siteNo)
indC = DF.varC.index(code)
c = DF.c[:, indS, indC]
t = DF.t

# subset
temp = np.where(~np.isnan(c))[0]
indT = temp[temp <= indT2]
np.random.seed(0)
np.random.shuffle(indT)
pLst = [100, 75, 50, 25]
for p in pLst:
    mask = np.ones([indT2-indT1+1, 1]).astype(bool)
    mask[indT[:int(len(indT)*p/100)], 0] = False
    subName = '{}-p{}'.format(siteNo, p)
    DF.saveSubset(subName, sd=str(t1), ed=str(t2),
                  siteNoLst=[siteNo], mask=mask)

# plot
# fig, ax = plt.subplots(1, 1)
# axplot.plotTS(ax, t, [c], cLst='k')
# for p, color in zip(pLst, 'rmgb'):
#     subName = '{}-p{}'.format(siteNo, p)
#     cc = DF.extractSubset(DF.c, subName)[:, 0, indC]
#     tt = DF.t[indT1:indT2+1]
#     axplot.plotTS(ax, tt, [cc], cLst=color)
# fig.show()

# train
label = 'QFPRT2C'
for p in pLst:
    trainSet = subName = '{}-p{}'.format(siteNo, 25)
    varX = dbBasin.label2var(label.split('2')[0])
    mtdX = dbBasin.io.extractVarMtd(varX)
    varY = dbBasin.label2var(label.split('2')[1])
    mtdY = dbBasin.io.extractVarMtd(varY)
    varXC = gageII.varLst
    mtdXC = dbBasin.io.extractVarMtd(varXC)
    varYC = None
    mtdYC = [code]
    outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
    dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                                 nEpoch=100, batchSize=[365, 20], nIterEp=50,
                                 varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                 mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
    basinFull.trainModel(outName)
