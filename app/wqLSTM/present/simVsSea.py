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


fig = plt.figure(figsize=(16, 5))
gs = gridspec.GridSpec(1, 31)
ngs = 10

for k, code in enumerate(codePlot):
    # code = codeLst[0]
    codeStr = usgs.codePdf.loc[code]['shortName']
    codeStr
    indC = DF.varC.index(code)
    indS = np.where(~matRm[:, indC])[0]
    axP = fig.add_subplot(gs[0, k*ngs:(k+1)*ngs])
    cs = axP.scatter(
        matQ[indS, indC], matS[indS, indC], c=matQS[indS, indC])
    axplot.titleInner(axP,codeStr)
    axP.set_xlabel('linearity')
    if k == 0:
        axP.set_ylabel('seasonality')
    else:
        axP.set_yticklabels([])
    axP.plot([0, 1], [0, 1], '-k')
    axP.set_aspect(1)

cax = fig.add_subplot(gs[0, -1])
cax = fig.colorbar(cs, cax=cax, orientation='vertical')
cax.set_label('simplicity')
fig.show()
