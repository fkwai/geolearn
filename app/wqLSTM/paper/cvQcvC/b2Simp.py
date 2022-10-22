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
from scipy.stats import linregress


# investigate CV(C) / CV(Q) as an indicator of model performance

codeLst = usgs.varC

DF = dbBasin.DataFrameBasin('G200')
# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)
matRm = count < 200

# CVc/CVq
matCV = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
matCV2 = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
matS = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
q = DF.q[:, :, 1]
cvQ = np.nanstd(q, axis=0)/np.nanmean(q, axis=0)
for k, code in enumerate(codeLst):
    c = DF.c[:, :, k]
    cvC = np.nanstd(c, axis=0)/np.nanmean(c, axis=0)
    for kk in range(len(DF.siteNoLst)):
        cc, qq = utils.rmNan([c[:, kk], q[:, kk]], returnInd=False)
        if len(cc) > 0:
            s, i, r, p, std_err = linregress(np.log(qq), np.log(cc))
            matS[kk, k] = s
            tc = np.nanstd(cc, axis=0)/np.nanmean(cc, axis=0)
            tq = np.nanstd(qq, axis=0)/np.nanmean(qq, axis=0)
            matCV2[kk, k] = tc/tq
    matCV[:, k] = cvC/cvQ
matCV[matRm] = np.nan
matCV2[matRm] = np.nan

# load linear/seasonal
dirParLst = [r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param',
             r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\Q\param',
             r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\S\param']
saveNameLst = ['QS', 'Q', 'S']
dictLR = dict()
for dirPar, saveName in zip(dirParLst, saveNameLst):
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirPar, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    dictLR[saveName] = matLR
matQ = dictLR['Q']


# count
for data in [matQ, matS, matCV2]:
    data[matRm] = np.nan

# Slope vs simp
codeCal = ['00095', '00915', '00925', '00930',
           '00935', '00940', '00945', '00955']
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(4, 2)
for k, code in enumerate(codeCal):
    j, i = utils.index2d(k, 4, 2)
    ax = fig.add_subplot(gs[j:j+1, i:i+1])
    ic = codeLst.index(code)
    ax.plot(matQ[:,ic], matS[:, ic], '*')
    ax.axhline(0,color='r')
    codeStr = usgs.codePdf.loc[code]['shortName']
    ax.set_title('{} {}'.format(code, codeStr))
plt.tight_layout()
fig.show()
