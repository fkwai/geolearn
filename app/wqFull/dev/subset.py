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
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)

siteNoTemp = DF.siteNoLst[:5]

importlib.reload(dbBasin)
DF.saveSubset('B10', ed='2009-12-31')
DF.saveSubset('A10', sd='2010-01-01')

# pick by year
yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(DF.t, yrIn)
t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
DF.createSubset('pkYr5', dateLst=t1)
DF.createSubset('rmYr5', dateLst=t2)

# pick by day
t1 = dbBasin.func.pickByDay(DF.t, dBase=5, dSel=1)
t2 = dbBasin.func.pickByDay(DF.t, dBase=5, dSel=1, pick=False)
DF.createSubset('pkD5', dateLst=t1)
DF.createSubset('rmD5', dateLst=t2)

# pick by random
t1 = dbBasin.func.pickRandT(DF.t, 0.2)
t2 = dbBasin.func.pickRandT(DF.t, 0.2, pick=False)
DF.createSubset('pkRT20', dateLst=t1)
DF.createSubset('rmRT20', dateLst=t2)

# plot
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
d1 = dbBasin.DataModelBasin(DF, subset='pkR20', varY=codeSel)
d2 = dbBasin.DataModelBasin(DF, subset='rmR20', varY=codeSel)

k = 0
fig, axes = figplot.multiTS(
    d2.t, [d2.Y[:, k, :], d1.Y[:, k, :]],  cLst='br', styLst='..')
fig.show()
