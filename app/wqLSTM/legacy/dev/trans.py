from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath
import json
import os
import importlib
importlib.reload(axplot)
importlib.reload(figplot)


dm = dbBasin.DataFrameBasin('weathering')

# subset
dm.saveSubset('B10', ed='2009-12-31')
dm.saveSubset('A10', sd='2010-01-01')


yrIn = np.arange(1985, 2020, 5).tolist()
t1 = dbBasin.func.pickByYear(dm.t, yrIn, pick=False)
t2 = dbBasin.func.pickByYear(dm.t, yrIn)
dm.createSubset('rmYr5', dateLst=t1)
dm.createSubset('pkYr5', dateLst=t2)

codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
d1 = dbBasin.DataModelBasin(dm, varY=codeSel, subset='rmYr5')
d2 = dbBasin.DataModelBasin(dm, varY=codeSel, subset='pkYr5')

mtdY = ['QT' for var in codeSel]
d1.trans(mtdY=mtdY)
d1.saveStat('temp')
# d2.borrowStat(d1)
d2.loadStat('temp')
yy = d2.y
yP = d2.transOutY(yy)
yO = d2.Y

# TS
indS = 1
fig, axes = figplot.multiTS(d1.t, [yO[:, indS, :], yP[:, indS, :]])
fig.show()

indS = 1
fig, axes = figplot.multiTS(d1.t, [yy[:, indS, :]])
fig.show()
