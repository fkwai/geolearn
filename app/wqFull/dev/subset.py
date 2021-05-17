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


# define a subset
d1 = dict(type='chunk', dateLst=[None, '2009-12-31'], siteNoLst=None)
# d2 = dict(name='pkYr5', type='slice',  dateLst=dateLst, siteNoLst=None)
d3 = dict(name='test', type='sparse')


importlib.reload(dbBasin)
DM.saveSubset('B10', dateLst=[None, '2009-12-31'])

yrIn = np.arange(1985, 2020, 5).tolist()
tOut = dbBasin.func.pickByYear(DM.t, yrIn)
DM.saveSubset('pkYr5',  dateLst=tOut)

dictSub = DM.loadSubset()
indT, indS, mask = DM.readSubset('pkYr5')

aa = dbBasin.DataTrain(DM, subset='B10')
