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

fig = plt.figure(figsize=[12, 6])
gs = gridspec.GridSpec(3, 3)

# seasonal
indS = np.where(matS == np.nanmax(matS))[0]
indC = np.where(matS == np.nanmax(matS))[1]
ax1 = fig.add_subplot(gs[0, 0:2])
axS1 = fig.add_subplot(gs[0, 2])
axT1 = ax1.twinx()
x = (DF.f[:, indS, DF.varF.index('tmmn')] +
     DF.f[:, indS, DF.varF.index('tmmx')])/2-273.15
q = DF.q[:, indS, 1]
c = DF.c[:, indS, indC]
axplot.plotTS(axT1, DF.t, x, cLst='b', lineW=[0.1])
axplot.plotTS(ax1, DF.t, c, cLst='r')
codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
ax1.set_title('{} {} linearity = {:.2f}, seasonality = {:.2f}'.format(
    DF.siteNoLst[indS[0]],codeStr, matQ[indS, indC], matS[indS, indC]))
sc = axS1.scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)

# linear
ax2 = fig.add_subplot(gs[1, 0:2])
axS2 = fig.add_subplot(gs[1, 2])
axT2 = ax2.twinx()
indS = np.where(matQ == np.nanmax(matQ))[0]
indC = DF.varC.index('00915')
[v], indV = utils.rmNan([matQ[:, indC]])
indSel = indV[np.argsort(-abs(v))[:10]]
indS = indSel[1]
q = DF.q[:, indS, 1]
c = DF.c[:, indS, indC]
axplot.plotTS(axT2, DF.t, q, cLst='b', lineW=[0.1])
axplot.plotTS(ax2, DF.t, c, cLst='r')
codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
ax2.set_title('{} {} linearity = {:.2f}, seasonality = {:.2f}'.format(
    DF.siteNoLst[indS[0]],codeStr, matQ[indS, indC], matS[indS, indC]))
sc = axS2.scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)

# both
ax3 = fig.add_subplot(gs[2, 0:2])
axS3 = fig.add_subplot(gs[2, 2])
axT3 = ax2.twinx()
lstS, lstC = np.where((matQ > 0.5) & (matS > 0.5))
k = 3
indS = lstS[k]
indC = lstC[k]
q = DF.q[:, indS, 1]
c = DF.c[:, indS, indC]
axplot.plotTS(ax3, DF.t, DF.c[:, indS, indC], cLst='r')
sc = axS3.scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)
codeStr = usgs.codePdf.loc[DF.varC[indC]]['shortName']
ax3.set_title('{} {} linearity = {:.2f}, seasonality = {:.2f}'.format(
    DF.siteNoLst[indS[0]],codeStr, matQ[indS, indC], matS[indS, indC]))
fig.show()
