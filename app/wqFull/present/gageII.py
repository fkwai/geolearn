from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y28N5.json')) as f:
    dictSite = json.load(f)

siteNoSel = dictSite['rmTK']
dfCrd = gageII.readData(siteNoLst=siteNoSel, varLst=[
                        'DRAIN_SQKM', 'LNG_GAGE', 'LAT_GAGE'])
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
area = dfCrd['DRAIN_SQKM'].values
fig, ax = plt.subplots(1, 1, figsize=(6, 8))
axplot.mapPoint(ax, lat, lon, area, s=16)
fig.show()
