

import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
codeLst = usgs.varC

DF = dbBasin.DataFrameBasin('G200')
# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)

matRm = count < 200

# load linear/seasonal
dirP = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\{}\param'
labLst = ['Q', 'S', 'QS']
dictS = dict()
for lab in labLst:
    dirS = dirP.format(lab)
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirS, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    dictS[lab] = matLR

matQ = dictS['Q']
matS = dictS['S']
matQS = dictS['QS']

fig, axes = plt.subplots(5, 4)
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 5, 4)
    codeStr = usgs.codePdf.loc[code]['shortName']
    ax = axes[j, i]
    # ax.scatter(matQS[:, k], matQ[:, k], color='none', edgecolor='b')
    # ax.scatter(matQS[:, k], matS[:, k], color='none', edgecolor='r')
    ax.scatter(matQ[:, k], matS[:, k], c=matQS[:, k])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], 'k-')
    if i > 0:
        ax.set_yticklabels([])
    if j < 4:
        ax.set_xticklabels([])
    axplot.titleInner(ax, '{} {}'.format(code, codeStr))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
fig.show()
