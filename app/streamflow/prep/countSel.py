from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import scipy

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
dfCrd = gageII.updateCode(dfCrd)

# plot
matCountQ = np.load(os.path.join(dirInv, 'matCountQ.npy'))
count1 = np.sum(matCountQ[:30], axis=0)/10958
count2 = np.sum(matCountQ[30:], axis=0)/3652
bRef = dfCrd['CLASS'].values == 1
rLst = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
nLst = list()
for r in rLst:
    # ind = np.where((count1 >= r) & (count2 >= r))[0]
    ind = np.where((count1 >= r) & (count2 >= r) & bRef)[0]
    nLst.append(len(ind))
fig, ax = plt.subplots(1, 1)
ax.plot(rLst, nLst)
fig.show()

# select 0.9
r = 0.9
# ind = np.where((count1 >= r) & (count2 >= r))[0]
ind = np.where((count1 >= r) & (count2 >= r) & bRef)[0]
siteNoLst = [siteNoLstAll[x] for x in ind]
dfSiteNo = pd.DataFrame(data=sorted(siteNoLst))
dfSiteNo.to_csv(os.path.join(dirInv, 'siteSel', 'Q90ref'),
                index=False, header=False)
