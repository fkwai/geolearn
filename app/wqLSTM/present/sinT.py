from hydroDL.data import ntn, dbBasin, usgs
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read time series
siteNo = '01184000'
df = dbBasin.readSiteTS(siteNo, dbBasin.io.varTLst,ed = np.datetime64('2010-01-01'))

fig, axes = plt.subplots(3, 1, figsize=(12, 6))
axplot.plotTS(axes[0], df.index, df['datenum'], styLst='-', cLst='k')
axplot.plotTS(axes[1], df.index, df['sinT'], styLst='-', cLst='k')
axplot.plotTS(axes[2], df.index, df['cosT'], styLst='-', cLst='k')
fig.show()


df = dbBasin.readSiteTS(siteNo, usgs.varC+usgs.varQ)

fig, axes = plt.subplots(3, 1, figsize=(12, 6))
axplot.plotTS(axes[0], df.index, df['runoff'], styLst='-', cLst='k')
axplot.plotTS(axes[1], df.index, df['00915'], styLst='*', cLst='k')
axplot.plotTS(axes[2], df.index, df['00945'], styLst='*', cLst='k')
fig.show()
