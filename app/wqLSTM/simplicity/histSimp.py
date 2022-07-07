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

np.nansum(matQS)

codePlot = ['00300', '00915', '00618']
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 1.5})
matplotlib.rcParams.update({'lines.markersize': 8})

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for k, code in enumerate(codePlot):
    # code = codeLst[0]
    codeStr = usgs.codePdf.loc[code]['shortName']
    codeStr
    indC = DF.varC.index(code)
    data = matQS[:, indC]
    indS = np.where(~np.isnan(data))
    s = data[indS]
    axes[k].hist(s, bins=np.arange(0, 1, 0.1), density=True)
    axes[k].set_title('{} #site={}'.format(codeStr, len(s)))
    axes[k].set_xlim(0, 1)
    axes[k].set_ylim(0, 2.5)
    axes[k].set_yticks([0, 1, 2])
    axes[k].set_yticklabels([0, 0.1, 0.2])
    fig.show()
