
import importlib
import numpy as np
import os
import pandas as pd
import json
from hydroDL.master import basins
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import usgs, gageII
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
codeLst = sorted(usgs.newC)

dirC = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-DS', 'All', 'params')
dirQ = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-DQ', 'All', 'params')

codeQ = '00060'
codeC = '00915'

fileParQ = os.path.join(dirQ, codeQ)
parQ = pd.read_csv(fileParQ, dtype={'siteNo': str}).set_index('siteNo')
fileParC = os.path.join(dirC, codeC)
parC = pd.read_csv(fileParC, dtype={'siteNo': str}).set_index('siteNo')

dfQ = parQ.loc[dictSite[codeC]]
dfC = parC.loc[dictSite[codeC]]


fig, axes = plt.subplots(1, 2)
axes[0].plot(dfQ['pSinT']/dfQ['pCosT'], dfC['pSinT']/dfC['pCosT'], '*')
axes[1].plot(dfQ['pCosT'], dfC['pCosT'], '*')
fig.show()

siteNo
t = np.arange(0, 10, 0.1)
pSinT=parQ
