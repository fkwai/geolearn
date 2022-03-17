import pickle
import hydroDL
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

dataName = 'weathering'
sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'weathering'
freq = 'D'
# DM = dbBasin.DataFrameBasin.new(
#     dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)
DF = dbBasin.DataFrameBasin(dataName)

DF.saveSubset('B10', ed='2009-12-31')
DF.saveSubset('A10', sd='2010-01-01')
yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(DF.t, yrIn)
t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
DF.createSubset('pkYr5', dateLst=t1)
DF.createSubset('rmYr5', dateLst=t2)

codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
d1 = dbBasin.DataModelBasin(DF, subset='rmYr5', varY=codeSel)
d2 = dbBasin.DataModelBasin(DF, subset='pkYr5', varY=codeSel)
print(type(DF) is hydroDL.data.dbBasin.DataModelBasin)


tempFolder = os.path.join(kPath.dirCode, 'temp')
mtdX = dbBasin.io.extractVarMtd(d1.v)


x = d1.X[:, :, 9]
dataIn = np.repeat(x[:, :, None], 6, axis=2)
mtdLst = ['norm', 'log-norm', 'stan', 'log-stan', 'QT', 'log-QT']


q = d1.X[:, :, -1]

# transIn
a = dataIn.copy()
b, dictTran = transform.transIn(a, mtdLst=mtdLst)
d, dictTran = transform.transIn(a, **dictTran)
c = transform.transOut(b, dictTran)

# check hist
nd = len(mtdLst)
bins = 50
indS = 8
fig, axes = plt.subplots(nd, 2, figsize=(4, 8))
for k, var in enumerate(mtdLst):
    _ = axes[k, 0].hist(a[:, :, k].flatten(), bins=bins)
    _ = axes[k, 1].hist(b[:, :, k].flatten(), bins=bins)
fig.show()

fig, axes = plt.subplots(nd, 2, figsize=(4, 8))
for k, var in enumerate(mtdLst):
    _ = axes[k, 0].hist(a[:, indS, k].flatten(), bins=bins)
    _ = axes[k, 1].hist(b[:, indS, k].flatten(), bins=bins)
fig.show()

# TS
fig, axes = figplot.multiTS(
    d1.t, b[:, indS, :], labelLst=mtdLst)
fig.show()

fig, axes = plt.subplots(nd, 2, figsize=(4, 8))
for k, var in enumerate(mtdLst):
    _ = axes[k, 0].hist(b[:, :, k].flatten(), bins=bins)
    _ = axes[k, 1].hist(b[:, :, k].flatten(), bins=bins)
    _ = axes[k, 0].hist(b[:, 8, k].flatten(), bins=bins)
    _ = axes[k, 1].hist(b[:, 11, k].flatten(), bins=bins)
fig.show()
