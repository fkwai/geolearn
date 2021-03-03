
import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

siteNoLst = dictSite['comb']

# load data
dfG = gageII.readData(siteNoLst=siteNoLst)
dfG = gageII.updateCode(dfG)
dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(
        siteNo, varLst=['00060'], freq='W')
    dictObs[siteNo] = df

varLst = ['CONTACT', 'ROCKDEPAVE', 'RockDEPAVE', 'DRAIN_SQKM']

t = dfG['CONTACT'].values
d = dfG['ROCKDEPAVE'].values
a = dfG['DRAIN_SQKM'].values
c = dfG['AWCAVE'].values

q = np.ndarray(len(siteNoLst))
for k, siteNo in enumerate(siteNoLst):
    q[k] = dictObs[siteNo]['00060'].mean()

t2 = d*a*c/q

dfG['RockDEPAVE'].max()

fig, ax = plt.subplots(1, 1)
ax.plot(t, t2, '*')
fig.show()
