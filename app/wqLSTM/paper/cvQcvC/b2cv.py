
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

code = '00300'
indC = DF.varC.index(code)
fig, ax = plt.subplots(1, 1)
ax.plot(matCV2[:, indC], matS[:, indC], '*')
ax.plot([0, 1], [0, 1], '-k')
ax.plot([0, 1], [0, -1], '-k')
ax.set_title('{} {}'.format(code, usgs.codePdf.loc[code]['shortName']))
fig.show()


fig, axes = plt.subplots(5, 4, figsize=(16, 10))
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
for k, code in enumerate(DF.varC):
    j, i = utils.index2d(k, 5, 4)
    axes[j, i].plot(matCV2[:, k], matS[:, k], '*')
    vM = np.nanmax(matCV2[:, k])
    axes[j, i].plot([0, vM], [0, vM], '-k')
    axes[j, i].plot([0, vM], [0, -vM], '-k')
    ns = np.sum(~matRm[:, k])
    codeStr = usgs.codePdf.loc[code]['shortName']
    titleStr = '{} #site={}'.format(codeStr, ns)
    axplot.titleInner(axes[j, i], titleStr, offset=0.15)
plt.tight_layout()
fig.show()
saveFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
fig.savefig(os.path.join(saveFolder, 'cv2b'))
fig.savefig(os.path.join(saveFolder, 'cv2b.svg'))
