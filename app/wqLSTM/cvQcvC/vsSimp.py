import matplotlib
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import sklearn.tree
import matplotlib.gridspec as gridspec
from hydroDL.master import basinFull


# investigate CV(C) / CV(Q) as an indicator of model performance

codeLst = usgs.varC

DF = dbBasin.DataFrameBasin('G200')
# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)
matRm = count < 200

matCV = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
q = DF.q[:, :, 1]
cvQ = np.nanstd(q, axis=0)/np.nanmean(q, axis=0)
for k, code in enumerate(codeLst):
    c = DF.c[:, :, k]
    cvC = np.nanstd(c, axis=0)/np.nanmean(c, axis=0)
    matCV[:, k] = cvC/cvQ

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan


codeGroup = [
    ['00010', '00300'],
    ['00915', '00925', '00930', '00955'],
    ['00600', '00605', '00618', '00660', '00665', '00681', '71846'],
    ['00095', '00400', '00405', '00935', '00940', '00945', '80154']
]
colorGroup = 'rmgb'
labGroup = ['stream', 'weathering', 'nutrient', 'mix']
a0 = matCV
b0 = matLR
a = np.nanmean(a0, axis=0)
b = np.nanmean(b0, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
for k in range(len(codeLst)):
    codeStr = usgs.codePdf.loc[codeLst[k]]['shortName']
    if codeStr in usgs.dictLabel.keys():
        ax.text(a[k], b[k], usgs.dictLabel[codeStr], fontsize=16)
    else:
        ax.text(a[k], b[k], codeStr, fontsize=16)
for codeG, colorG, labG in zip(codeGroup, colorGroup, labGroup):
    ind = [codeLst.index(code) for code in codeG]
    ax.plot(a[ind], b[ind], color=colorG, label=labG, marker='o', ls='None')
    for k in ind:
        aa = [np.nanpercentile(a0[:, k], 25), np.nanpercentile(a0[:, k], 75)]
        bb = [np.nanpercentile(b0[:, k], 25), np.nanpercentile(b0[:, k], 75)]
        ax.plot([a[k], a[k]], bb, color=colorG,
                linestyle='dashed', linewidth=0.5)
        ax.plot(aa, [b[k], b[k]], color=colorG,
                linestyle='dashed', linewidth=0.5)
# ax.axhline(0, color='k')
# ax.axvline(0.4, color='k')
ax.set_xlabel('CVc / CVq')
ax.set_ylabel('simplicity')
fig.show()

# calculate a coefficient
codeCal = [
    '00915', '00925', '00930', '00955', '00600','00030'
    '00605', '00618', '00660', '00665', '00681', '71846',
    '00095',  '00935', '00940', '00945'
]
ind = [codeLst.index(code) for code in codeCal]
np.corrcoef(a[ind], b[ind])
