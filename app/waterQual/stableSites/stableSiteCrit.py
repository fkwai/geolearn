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
import json

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
dfCrd = gageII.updateCode(dfCrd)
# countMatD = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMatW = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))


# ny years > ns samples summary
indLst = list()
ny = 2
nsLst = range(5, 30)
outMat = np.ndarray([len(nsLst), len(codeLst)+2])
for ks, ns in enumerate(nsLst):
    indLst = list()
    print('counting ny>{}, ns>{}'.format(ny, ns))
    for kc, code in enumerate(codeLst):
        ic = codeLst.index(code)
        count = countMatW[:, :, ic]
        indS = np.where(np.sum(count > ns, axis=1) > ny)[0]
        outMat[ks, kc] = len(indS)
        indLst.append(indS)
    # get rid of 00010 and 00095
    # indAll = np.unique(np.concatenate(indLst[2:]))
    outMat[ks, -2] = len(np.unique(np.concatenate(indLst)))
    outMat[ks, -1] = len(np.unique(np.concatenate(indLst[2:])))

# plot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for kc, code in enumerate(codeLst):
    ax.plot(nsLst,outMat[:, kc], '-*', label=codeLst[kc])
ax.plot(nsLst,outMat[:, -1], '-*', label='comb1')
ax.plot(nsLst,outMat[:, -2], '-*', label='comb2')
ax.legend()
fig.show()


# # plot a hitmap - hard to interpret
# code = '00600'
# ic = codeLst.index(code)
# count = countMatW[:, :, ic]
# nyLst = list(range(20))
# nsLst = list(range(20))
# mat = np.ndarray([len(nsLst), len(nyLst)])
# for j, ns in enumerate(nsLst):
#     for i, ny in enumerate(nyLst):
#         mat[j, i] = len(np.where(np.sum(count > ns, axis=1) > ny)[0])
# # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# # axplot.plotHeatMap(ax, mat, labLst=list(range(20)))
# # fig.show()
# grad = np.gradient(mat)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# axplot.plotHeatMap(ax, (grad[0]+grad[1])/2/mat*100, labLst=list(range(20)))
# ax.set_title(code)
# # axplot.plotHeatMap(ax, (grad[1])/mat*100, labLst=list(range(20)))
# fig.show()
