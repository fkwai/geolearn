from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import scipy
import json

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
countMatD = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMatW = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))


ny = 3
nsLst = np.arange(5, 20)*ny
# nsLst = [20, 24, 28, 32, 36, 40, 44, 45,
#          46, 47, 48, 52, 56, 60, 64, 68, 72, 76]
outMat = np.ndarray([len(codeLst), len(nsLst)])
for i, code in enumerate(codeLst):
    ic = codeLst.index(code)
    count = np.sum(countMatW[:, -ny:, ic], axis=1)
    for j, ns in enumerate(nsLst):
        outMat[i, j] = np.sum(count >= ns)

# plot
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
axplot.plotHeatMap(ax, outMat, labLst=[codeLst, nsLst])
fig.show()
