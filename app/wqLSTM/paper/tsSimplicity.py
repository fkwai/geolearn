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


td = pd.to_datetime(DF.t).dayofyear
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})

fig = plt.figure(figsize=[12, 10])
gs = gridspec.GridSpec(3, 3)

# seasonal
ax1 = fig.add_subplot(gs[0, 0:2])
axS1 = fig.add_subplot(gs[0, 2])
axT1 = ax1.twinx()
indC = DF.varC.index('00300')
siteNo = '04137500'
indS = DF.siteNoLst.index(siteNo)
codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
x = (DF.f[:, indS, DF.varF.index('tmmn')] +
     DF.f[:, indS, DF.varF.index('tmmx')])/2-273.15
q = DF.q[:, indS, 1]
c = DF.c[:, indS, indC]
axT1.invert_yaxis()
axT1.plot(DF.t, x, 'r-', label='Temperature', linewidth=0.1)
ax1.plot(DF.t, c, 'k*', label=codeStr)
axT1.tick_params(axis='y', colors='r')

codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
ax1.set_title('{} at {}; simplicity = {:.2f}; linearity = {:.2f}, seasonality = {:.2f}'.format(
    codeStr, siteNo, matQS[indS, indC], matQ[indS, indC], matS[indS, indC]))
sc = axS1.scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)
ax1.set_xticklabels([])

# linear
ax2 = fig.add_subplot(gs[1, 0:2])
axS2 = fig.add_subplot(gs[1, 2])
axT2 = ax2.twinx()
indC = DF.varC.index('00915')
siteNo = '04063700'
indS = DF.siteNoLst.index(siteNo)
q = DF.q[:, indS, 1]
c = DF.c[:, indS, indC]
axT2.invert_yaxis()
axT2.plot(DF.t, q, 'b-', label='Streamflow', linewidth=0.1)
ax2.plot(DF.t, c, 'k*', label=codeStr)
axT2.tick_params(axis='y', colors='b')

codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
ax2.set_title('{} at {}; simplicity = {:.2f}; linearity = {:.2f}, seasonality = {:.2f}'.format(
    codeStr, siteNo, matQS[indS, indC], matQ[indS, indC], matS[indS, indC]))
sc = axS2.scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)
ax2.set_xticklabels([])

# both
ax3 = fig.add_subplot(gs[2, 0:2])
axS3 = fig.add_subplot(gs[2, 2])
axT3 = ax3.twinx()
axT4 = ax3.twinx()
indC = DF.varC.index('00930')
siteNo = '11264500'
indS = DF.siteNoLst.index(siteNo)
q = DF.q[:, indS, 1]
x = (DF.f[:, indS, DF.varF.index('tmmn')] +
     DF.f[:, indS, DF.varF.index('tmmx')])/2-273.15
c = DF.c[:, indS, indC]
axT3.invert_yaxis()
axT4.invert_yaxis()
axT3.plot(DF.t, q, 'b-', label='Streamflow', linewidth=0.1)
axT4.plot(DF.t, x, 'r-', label='Temperature', linewidth=0.1)
ax3.plot(DF.t, c, 'k*', label=codeStr)
axT3.tick_params(axis='y', colors='b')
axT4.tick_params(axis='y', colors='r')
sc = axS3.scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)
codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
ax3.set_title('{} at {}; simplicity = {:.2f}; linearity = {:.2f}, seasonality = {:.2f}'.format(
    codeStr, siteNo, matQS[indS, indC], matQ[indS, indC], matS[indS, indC]))
fig.show()

outFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
fig.savefig(os.path.join(outFolder, 'tsSimplicity'))
