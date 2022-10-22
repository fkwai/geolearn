
from logging import raiseExceptions
import numpy as np
from numpy import double
import pandas as pd
import os
import hydroDL
from hydroDL.data import camels, usgs, gridMET
import importlib
import matplotlib.pyplot as plt
from hydroDL.post import axplot
from hydroDL.utils.time import t2dt


importlib.reload(camels)
dfInfo = camels.dfInfo

dirDB = r'C:\Users\geofk\work\database\camels'
siteNo = '01013500'
df1 = camels.readStreamflow(siteNo)
df2 = usgs.readStreamflow(siteNo)

fig, ax = plt.subplots(1, 1)
ax.plot(df1.index, df1['q'], '-b', label='camels')
ax.plot(df2.index, df2['00060_00003'], '-r', label='usgs')
ax.legend()
ax.set_xlim([t2dt(20000101), t2dt(20040101)])
fig.show()

dfF0 = gridMET.readBasin(siteNo)
dfF1 = camels.readForcing(siteNo, opt='nldas')
dfF2 = camels.readForcing(siteNo, opt='maurer')
dfF3 = camels.readForcing(siteNo, opt='daymet')


fig, ax = plt.subplots(1, 1)
ax.plot(dfF0.index, dfF0['pr'], '--*k', label='gridMet')
ax.plot(dfF1.index, dfF1['prcp'], '--*r', label='nldas')
ax.plot(dfF2.index, dfF2['prcp'], '--*g', label='maurer')
ax.plot(dfF3.index, dfF3['prcp'], '--*b', label='daymet')
ax.legend()
# ax.set_xlim([t2dt(20000101), t2dt(20040101)])
fig.show()


dfF1.columns
dfF2.columns
