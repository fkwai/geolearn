import pandas as pd
from hydroDL.data import dbBasin, usgs
import json
import os
from hydroDL import kPath
import numpy as np
from hydroDL.post import axplot, figplot
import random

# DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, sdStr=sd, edStr=ed)
dataName = 'rmTK-B200'
DF = dbBasin.DataFrameBasin(dataName)

# randomly 5-fold
seed = 0
t = DF.t
indT = np.arange(len(DF.t))
random.Random(seed).shuffle(indT)
for k in range(5):
    t1 = np.sort(t[indT[::5]])
    t2 = np.setdiff1d(t, t1)
    DF.createSubset('pkRT5b{}'.format(k), dateLst=t1)
    DF.createSubset('rmRT5b{}'.format(k), dateLst=t2)


# pick by year
sy = DF.t[0].astype(object).year
ey = DF.t[-1].astype(object).year
yrAry = np.array(range(sy, ey))
for k in range(5):
    yrIn = yrAry[yrAry % 5 == k]
    t1 = dbBasin.func.pickByYear(DF.t, yrIn)
    t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
    DF.createSubset('pkYr5b{}'.format(k), dateLst=t1)
    DF.createSubset('rmYr5b{}'.format(k), dateLst=t2)

# before after 2015
DF.saveSubset('B15', ed='2015-12-31')
DF.saveSubset('A15', sd='2016-01-01')

# examine test/train rate
trainLst = ['rmYr5b0', 'rmRT5b0', 'B15']
testLst = ['pkYr5b0', 'pkRT5b0', 'A15']
df = pd.DataFrame(index=DF.varC, columns=trainLst)
aLst = list()
bLst = list()
for trainSet, testSet in zip(trainLst, testLst):
    a = DF.extractSubset(DF.c, trainSet)
    b = DF.extractSubset(DF.c, testSet)
    aLst.append(a)
    bLst.append(b)

dataBox = list()
for code in DF.varC:
    indC = DF.varC.index(code)
    temp = list()
    for trainSet, a, b in zip(trainLst, aLst, bLst):
        x = ~np.isnan(a[:, :, indC])
        y = ~np.isnan(b[:, :, indC])
        n1 = np.sum(x, axis=0)
        n2 = np.sum(y, axis=0)
        indS = np.where((n1 > 160) & (n2 > 40))[0]
        temp.append(n2[indS] / n1[indS])
        # df.at[code, trainSet] = len(indS)
        # temp.append(n1[indS])
    dataBox.append(temp)
labLst1 = [
    '{}\n{}'.format(usgs.codePdf.loc[code]['shortName'], code) for code in DF.varC
]
fig, ax = figplot.boxPlot(dataBox, label1=labLst1, figsize=(6, 4))
fig.show()
