from mpl_toolkits import basemap
from hydroDL.data import gageII
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from hydroDL.post import axplot

regionLst = [
    'ECO2_BAS_DOM',
    'NUTR_BAS_DOM',
    'HLR_BAS_DOM_100M',
    'PNV_BAS_DOM',
]
dfG = gageII.readData(varLst=regionLst+['LAT_GAGE', 'LNG_GAGE', 'CLASS'])

# deal with PNV
fileT = os.path.join(gageII.dirTab, 'lookupPNV.csv')
tabT = pd.read_csv(fileT).set_index('PNV_CODE')
for code in range(1, 63):
    siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
    dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']

# load selected sites
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# plot map
# var = 'PNV_BAS_DOM2'
var = 'ECO2_BAS_DOM'
lat = dfG.loc[siteNoLst]['LAT_GAGE']
lon = dfG.loc[siteNoLst]['LNG_GAGE']
data = dfG.loc[siteNoLst][var]
fig, ax = plt.subplots(1, 1)
mm = basemap.Basemap(llcrnrlat=25, urcrnrlat=50,
                     llcrnrlon=-125, urcrnrlon=-65,
                     projection='cyl', resolution='c', ax=ax)
mm.drawcoastlines()
mm.drawcountries(linestyle='dashed')
mm.drawstates(linestyle='dashed', linewidth=0.5)
dataUni = np.unique(data)
cmap = plt.cm.jet
cAry = cmap(np.linspace(0, 1, len(dataUni)))
np.random.shuffle(cAry)
# dataUni=[16,20]
for k, dd in enumerate(dataUni):
    ind = np.where(data == dd)[0]
    cs = mm.scatter(lon[ind], lat[ind], color=cAry[k,:],
                    marker='o', label='{} {}'.format(dd, len(ind)))
ax.legend()
fig.show()
