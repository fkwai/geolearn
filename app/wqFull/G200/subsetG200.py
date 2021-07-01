
import pandas as pd
from hydroDL.data import dbBasin, usgs
import json
import os
from hydroDL import kPath
import numpy as np
from hydroDL.post import axplot, figplot

sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'G200'
dictSiteName = 'dict{}.json'.format(dataName)
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['rmTK']
# DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, sdStr=sd, edStr=ed)
DF = dbBasin.DataFrameBasin(dataName)

# print site count
for key in dictSite.keys():
    print(key, len(dictSite[key]))


seed = 0
rate = 0.2
rng = np.random.default_rng(seed)

# random subset
mask = np.ones([len(DF.t), len(DF.siteNoLst)]).astype(bool)
for indS, siteNo in enumerate(DF.siteNoLst):
    obsB = np.any(~np.isnan(DF.c[:, indS, :]), axis=1)
    obsD = np.where(obsB)[0]
    nPick = int(sum(obsB*rate))
    ind = rng.choice(obsD, nPick, replace=False)
    mask[ind, indS] = False
DF.saveSubset('pkR20', mask=mask)
DF.saveSubset('rmR20', mask=~mask)


# last 20% subset
mask = np.ones([len(DF.t), len(DF.siteNoLst)]).astype(bool)
for indS, siteNo in enumerate(DF.siteNoLst):
    obsB = np.any(~np.isnan(DF.c[:, indS, :]), axis=1)
    obsD = np.where(obsB)[0]
    nPick = int(sum(obsB*rate))
    ind = obsD[-nPick:]
    mask[ind, indS] = False
DF.saveSubset('pkL20',  mask=mask)
DF.saveSubset('rmL20',  mask=~mask)

# pick by year
yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(DF.t, yrIn)
t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
DF.createSubset('pkYr5', dateLst=t1)
DF.createSubset('rmYr5', dateLst=t2)

# before after 2010
DF.saveSubset('B10', ed='2009-12-31')
DF.saveSubset('A10', sd='2010-01-01')

# examine test/train rate
trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']
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
        # temp.append(n2[indS]/n1[indS])
        df.at[code, trainSet] = len(indS)
        temp.append(n1[indS])
    dataBox.append(temp)
labLst1 = ['{}\n{}'.format(usgs.codePdf.loc[code]
                           ['shortName'], code) for code in DF.varC]
fig, ax = figplot.boxPlot(dataBox, label1=labLst1,
                          figsize=(6, 4))
fig.show()
