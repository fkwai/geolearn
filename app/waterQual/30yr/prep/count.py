from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import time
import json

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')

# countMat = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMat = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))

# count for obs before / after 2010
count1 = np.ndarray([len(siteNoLstAll), len(codeLst)])
count2 = np.ndarray([len(siteNoLstAll), len(codeLst)])
for ic, code in enumerate(codeLst):
    count1[:, ic] = np.sum(countMat[:, :30, ic], axis=1)
    count2[:, ic] = np.sum(countMat[:, 30:, ic], axis=1)

# plot
nsLst = np.arange(1, 20)
outMat = np.ndarray([len(codeLst), len(nsLst)])
for j, ns in enumerate(nsLst):
    outMat[:, j] = np.sum((count1 >= ns*30) & (count2 >= ns*10), axis=0)
    # outMat[:, j] = np.sum(count2 >= ns, axis=0)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
axplot.plotHeatMap(ax, outMat, labLst=[codeLst, nsLst])
fig.show()

# get siteNoLst
ns = 10
pickMat = (count1 >= ns*30) & (count2 >= ns*10)
siteNoLst = list(np.array(siteNoLstAll)[np.any(pickMat[:, 2:], axis=1)])
len(siteNoLst)
siteNoLst = list(np.array(siteNoLstAll)[np.any(pickMat, axis=1)])
len(siteNoLst)

# save for each code and comb
dictSite = dict()
indS = np.where(np.any(pickMat, axis=1))[0]
dictSite['comb'] = [siteNoLstAll[ind] for ind in indS]
indS = np.where(np.any(pickMat[:, 1:], axis=1))[0]
dictSite['combRmT'] = [siteNoLstAll[ind] for ind in indS]
indS = np.where(np.any(pickMat[:, 2:], axis=1))[0]
dictSite['combRmTK'] = [siteNoLstAll[ind] for ind in indS]
for code in codeLst:
    ic = codeLst.index(code)
    indS = np.where(pickMat[:, ic])[0]
    dictSite[code] = [siteNoLstAll[ind] for ind in indS]
saveName = os.path.join(dirInv, 'siteSel', 'dictRB_Y30N{}'.format(ns))
with open(saveName+'.json', 'w') as fp:
    json.dump(dictSite, fp, indent=4)
