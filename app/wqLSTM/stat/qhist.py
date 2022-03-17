
import scipy
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# load all site counts
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
countD = np.load(os.path.join(dirInv, 'matCountDaily.npy'))

code = '00915'
indC = codeLst.index(code)
count = np.sum(countD[:, :, indC], axis=1)
indSLst = np.where(count > 200)[0]
siteNoLst = [siteNoLstAll[ind] for ind in indSLst]

# DF = dbBasin.DataFrameBasin.new('00915G200', siteNoLst)
DF = dbBasin.DataFrameBasin('00915G200')
q = DF.q[:, :, 1]
c = DF.c[:, :, DF.varC.index(code)]

ns = len(DF.siteNoLst)
out = np.ndarray([ns, 2])
for indS in range(ns):
    q1 = np.log(q[:, indS]+1)
    ind = np.where(~np.isnan(c[:, indS]))[0]
    q2 = np.log(q[ind, indS]+1)
    s, p = scipy.stats.ks_2samp(q1, q2)
    out[indS, 0] = s
    out[indS, 1] = len(ind)

dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values

fig, axes = plt.subplots(2, 1)
axplot.mapPoint(axes[0], lat, lon, out[:, 0], s=16)
axplot.mapPoint(axes[1], lat, lon, out[:, 1], s=16)
fig.show()


# Q of Nine random sites
indLst = np.random.randint(0, ns, 9)
nfy, nfx = (3, 3)
bins = 10
fig, axes = plt.subplots(nfy, nfx)
for k, indS in enumerate(indLst):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = DF.varC.index(code)
    q1 = np.log(q[:, indS]+1)
    ind = np.where(~np.isnan(c[:, indS]))[0]
    q2 = np.log(q[ind, indS]+1)
    axplot.plotCDF(ax, [q1, q2])
    # ax.hist([q1, q2], bins=bins, density=True)
    nData = np.sum(~np.isnan(DF.c[:, indS, ic]))
    shortName = usgs.codePdf.loc[code]['shortName']
    titleStr = '{} {}'.format(
        DF.siteNoLst[indS], nData)
    axplot.titleInner(ax, titleStr, top=False)
fig.show()

q1 = np.log(q+1).flatten()
b = ~np.isnan(c)
q2 = np.log(q[b]+1)
fig, axes = plt.subplots(2, 1)
axplot.plotCDF(axes[0], [q1, q2])
axes[1].hist([q1, q2], bins=20, density=True)
fig.show()
s, p = scipy.stats.ks_2samp(q1, q2)