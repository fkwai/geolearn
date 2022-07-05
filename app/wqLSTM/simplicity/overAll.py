import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import sklearn.tree
import matplotlib.gridspec as gridspec


# investigate correlation between simlicity and basin attributes.
# remove carbon - less obs, high corr

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
dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200\simplicity'

# maps
lat, lon = DF.getGeo()
for lab in ['Q', 'S', 'QS']:
    mat = dictS[lab]
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(5, 4)
    for k, code in enumerate(codeLst):
        j, i = utils.index2d(k, 5, 4)
        ax = mapplot.mapPoint(fig, gs[j:j+1, i:i+1], lat, lon,
                              mat[:, k], cb=True)
        codeStr = usgs.codePdf.loc[code]['shortName']
        ax.set_title('{} {}'.format(code, codeStr))
    plt.tight_layout()
    fig.show()
    fig.savefig(os.path.join(dirPaper, 'mapSim_{}'.format(lab)))


# scatter
fig, axes = plt.subplots(5, 4, figsize=(16, 12))
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 5, 4)
    codeStr = usgs.codePdf.loc[code]['shortName']
    ax = axes[j, i]
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
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(dirPaper, 'Q2S'))

# scatter normalized
fig, axes = plt.subplots(5, 4, figsize=(16, 12))
for k, code in enumerate(codeLst):
    j, i = utils.index2d(k, 5, 4)
    codeStr = usgs.codePdf.loc[code]['shortName']
    ax = axes[j, i]
    ax.scatter(matQ[:, k]/matQS[:, k], matS[:, k]/matQS[:, k], c=matQS[:, k])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], 'k-')
    if i > 0:
        ax.set_yticklabels([])
    if j < 4:
        ax.set_xticklabels([])
    axplot.titleInner(ax, '{} {}'.format(code, codeStr))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(dirPaper, 'Q2Snorm'))
