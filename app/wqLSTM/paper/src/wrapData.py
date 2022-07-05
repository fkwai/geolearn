
from hydroDL.data import dbBasin
import json
import os
from hydroDL import kPath
import numpy as np
import pandas as pd

sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'N200'
dictSiteName = 'dictG200.json'
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['rmTK']
DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, sdStr=sd, edStr=ed)
# DF = dbBasin.DataFrameBasin(dataName)

# normalization
# DFN = dbBasin.func.localNorm(DF, subset='all')
# DFN.saveAs(dataName+'N')


seed = 0
rate = 0.2
rng = np.random.default_rng(seed)

# random 20%
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

# random date 20%
t1 = dbBasin.func.pickRandT(DF.t, 0.2)
t2 = dbBasin.func.pickRandT(DF.t, 0.2, pick=False)
DF.createSubset('pkRT20', dateLst=t1)
DF.createSubset('rmRT20', dateLst=t2)

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
trainLst = ['rmYr5', 'rmR20', 'rmL20', 'rmRT20',  'B10']
testLst = ['rmYr5', 'pkR20', 'pkL20', 'pkRT20',  'A10']
df = pd.DataFrame(index=DF.varC, columns=trainLst)
aLst = list()
bLst = list()
for trainSet, testSet in zip(trainLst, testLst):
    a = DF.extractSubset(DF.c, trainSet)
    b = DF.extractSubset(DF.c, testSet)
    aLst.append(a)
    bLst.append(b)
