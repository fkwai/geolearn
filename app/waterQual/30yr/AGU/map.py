import scipy
import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from mpl_toolkits import basemap

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
dfCrd1 = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLst)
dfCrd1 = gageII.updateCode(dfCrd1)


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSiteN5 = json.load(f)
with open(os.path.join(dirSel, 'dictRB_Y30N2.json')) as f:
    dictSiteN2 = json.load(f)
codeLst = sorted(usgs.newC)
dictSite = dict()
for code in usgs.newC+['comb']:
    siteNoCode = list(set(dictSiteN2[code])-set(dictSiteN5['comb']))
    dictSite[code] = siteNoCode
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
dfCrd2 = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLst)
dfCrd2 = gageII.updateCode(dfCrd2)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
mm = basemap.Basemap(llcrnrlat=25, urcrnrlat=50,
                     llcrnrlon=-125, urcrnrlon=-65,
                     projection='cyl', resolution='c', ax=ax)
mm.drawcoastlines()
mm.drawcountries(linestyle='dashed')
mm.drawstates(linestyle='dashed', linewidth=0.5)
mm.plot(dfCrd1['LNG_GAGE'], dfCrd1['LAT_GAGE'], '*b', label='train basins')
mm.plot(dfCrd2['LNG_GAGE'], dfCrd2['LAT_GAGE'], '.r', label='test basins')
ax.legend()
fig.show()
