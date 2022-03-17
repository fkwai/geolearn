from hydroDL.data import ntn, dbBasin
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dfSite = ntn.loadSite()

lat = dfSite['latitude']
lon = dfSite['longitude']
yr = pd.to_datetime(dfSite['startdate']).values.astype(
    'M8[Y]').astype(str).astype(int)
fig, ax = plt.subplots(1, 1, figsize=(6, 8))
axplot.mapPoint(ax, lat, lon, yr, s=16)
fig.show()

# read time series
siteNo = '01184000'
df = dbBasin.readSiteTS(siteNo, ntn.varLst)

fig, axes = plt.subplots(2, 1, figsize=(6, 8))
axplot.plotTS(axes[0], df.index, df['Ca'], styLst='-', cLst='k')
axplot.plotTS(axes[1], df.index, df['distNTN'], styLst='-', cLst='k')
fig.show()
