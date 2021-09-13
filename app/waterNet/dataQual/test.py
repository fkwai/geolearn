import random
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

DF = dbBasin.DataFrameBasin('G400')

siteNoLst = DF.siteNoLst
codeLst = DF.varC
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM'],
                        siteNoLst=siteNoLst)
t1 = np.datetime64('1982-01-01')
t2 = np.datetime64('2010-01-01')
t3 = np.datetime64('2019-01-01')
indT1 = np.where(DF.t == t1)[0][0]
indT2 = np.where(DF.t == t2)[0][0]
indT3 = np.where(DF.t == t3)[0][0]

siteNo = '01434025'
code = '00915'
indS = DF.siteNoLst.index(siteNo)
indC = DF.varC.index(code)
c = DF.c[:, indS, indC]
t = DF.t

# select indT
temp = np.where(~np.isnan(c))[0]
indT = temp[temp <= indT2]
np.random.seed(0)
np.random.shuffle(indT)

pLst = [100, 75, 50, 25]
for p in pLst:
    mask = np.ones([indT2-indT1, 1]).astype(bool)
    mask[indT[:int(len(indT)*p/100)], 0] = False
    subName = '{}-p{}'.format(siteNo, p)
    DF.saveSubset(subName, sd=str(t1), ed=str(t2),
                  siteNoLst=[siteNo], mask=mask)


fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, t, [c], cLst='k')
for p, color in zip(pLst, 'rmgb'):
    subName = '{}-p{}'.format(siteNo, p)
    cc = DF.extractSubset(DF.c, subName)[:, 0, indC]
    tt = DF.t[indT1:indT2]
    # axplot.plotTS(ax, tt, [cc], cLst=color, markersize=p/2.5, marker='o')
    axplot.plotTS(ax, tt, [cc], cLst=color)
fig.show()
