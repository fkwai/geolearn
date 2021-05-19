from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
importlib.reload(axplot)
importlib.reload(figplot)

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
DM = dbBasin.DataModelFull.new(
    dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)

siteNoTemp = DM.siteNoLst[:5]

importlib.reload(dbBasin)
DM.saveSubset('B10', ed='2009-12-31')
DM.saveSubset('A10', sd='2010-01-01')

yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(DM.t, yrIn)
t2 = dbBasin.func.pickByYear(DM.t, yrIn, pick=False)

DM.createSubset('pkYr5', dateLst=t1)
DM.createSubset('rmYr5', dateLst=t2)

DM.createSubset('test', dateLst=t2, sd='2000-01-01',
                ed='2010-01-01', siteNoLst=siteNoTemp)


dictSub = DM.loadSubset()
indT1, indT2, indS, mask = DM.readSubset('rmYr5')

d1 = dbBasin.DataTrain(DM, subset='pkYr5')
d2 = dbBasin.DataTrain(DM, subset='rmYr5')

k = 0
fig, axes = figplot.multiTS(
    d2.t, [d1.Y[:, k, :], d2.Y[:, k, :]],  cLst='rb')
fig.show()


# aa = np.ndarray([5, 3])
