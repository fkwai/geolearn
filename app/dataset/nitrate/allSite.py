import cartopy.feature as cfeature
import random
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

codeLst = ['00400', '00600', '00605', '00618', '71846']
# DF = dbBasin.DataFrameBasin.new(
#     'allN', siteNoLstAll, varC=codeLst, varF=[], varQ=usgs.varQ, varG=gageII.varLst)
DF = dbBasin.DataFrameBasin('allN')

# count of obs
count = np.sum(~np.isnan(DF.c), axis=0)
fig, ax = plt.subplots(1, 1)
cLst = 'krgbc'
labLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
ns = count.shape[0]
for k in range(5):
    x = np.sort(count[:, k])[::-1]
    ax.plot(np.arange(1, ns+1), x, color=cLst[k], label=labLst[k])
ax.set_xlim([-50, ns])
# ax.set_ylim([0, np.max(count)])
ax.legend()
fig.show()
ax.set_yscale('log')
fig.show()
# ax.set_xlim([0, 1000])
# ax.set_ylim([0, 500])
# fig.show()

# values of obs
codePlot = ['00600', '00605', '00618', '71846']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for kk, code in enumerate(codePlot):
    k = codeLst.index(code)
    cs = usgs.codePdf.loc[code]['shortName']
    j, i = utils.index2d(kk, 2, 2)
    ax = axes[j, i]
    x = DF.c[:, :, k]
    v = x[~np.isnan(x)]
    ax.hist(v, bins=range(100), log=True)
    # ax.hist(v, bins=range(100))
    ax.set_ylabel('# samples')
    ax.set_xlabel('{} [mg/l]'.format(cs))
    ax.set_xlim([0, 40])
    ax.axvline(10, color='k')
fig.show()


# values of basins
codePlot = ['00600', '00605', '00618', '71846']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for kk, code in enumerate(codePlot):
    k = codeLst.index(code)
    cs = usgs.codePdf.loc[code]['shortName']
    j, i = utils.index2d(kk, 2, 2)
    ax = axes[j, i]
    x = DF.c[:, :, k]
    b = count[:, k] >= 50
    # v = np.nanmedian(x, axis=0)
    v = np.nanpercentile(x, 75, axis=0)
    ax.hist(v[b], bins=np.arange(0, 20, 0.5))
    ax.set_ylabel('# sites')
    # ax.set_xlabel('median {} [mg/l] of sites'.format(cs))
    ax.set_xlabel('75% {} [mg/l] of sites'.format(cs))
    ax.set_xlim([0, 20])
    ax.axvline(10, color='k')
fig.show()

# maps
latA, lonA = DF.getGeo()
figM = plt.figure()
gsM = gridspec.GridSpec(2, 2)
for kk, code in enumerate(codePlot):
    j, i = utils.index2d(kk, 2, 2)
    k = codeLst.index(code)
    cs = usgs.codePdf.loc[code]['shortName']
    x = DF.c[:, :, k]
    b = count[:, k] > 50
    v = np.nanmedian(x, axis=0)
    axM = mapplot.mapPoint(
        figM, gsM[j, i], latA[b], lonA[b], v[b], s=10, cb=True)
    axM.set_title('median {} [mg/l]'.format(cs))
figM.show()

# map with states
provinces_50m = cfeature.NaturalEarthFeature(
    'cultural',
    'admin_1_states_provinces_lines',
    '50m',
    facecolor='none')
code = '00618'
k = codeLst.index(code)
cs = usgs.codePdf.loc[code]['shortName']
x = DF.c[:, :, k]
b = count[:, k] > 50
v = np.nanmedian(x, axis=0)
figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(
    figM, gsM[0, 0], latA[b], lonA[b], v[b], s=10, cb=True)
axM.add_feature(provinces_50m)
axM.add_feature(cfeature.BORDERS)
axM.add_feature(cfeature.RIVERS)
axM.add_feature(cfeature.LAKES)
# axM = mapplot.mapPoint(
#     figM, gsM[0, 0], latA[b], lonA[b], v[b], s=10, cb=True)
axM.set_title('median {} [mg/l]'.format(cs))
figM.show()


# std
figM = plt.figure()
gsM = gridspec.GridSpec(2, 2)
for kk, code in enumerate(codePlot):
    j, i = utils.index2d(kk, 2, 2)
    k = codeLst.index(code)
    cs = usgs.codePdf.loc[code]['shortName']
    x = DF.c[:, :, k]
    b = count[:, k] > 50
    v = np.nanstd(x, axis=0)/np.nanmean(x, axis=0)
    axM = mapplot.mapPoint(
        figM, gsM[j, i], latA[b], lonA[b], v[b], s=10, cb=True)
    axM.set_title('CV {} [mg/l]'.format(cs))
figM.show()

# std vs median

figM = plt.figure()
gsM = gridspec.GridSpec(3, 1)
r1 = DF.c[:, :, 2]/DF.c[:, :, 1]
r2 = DF.c[:, :, 3]/DF.c[:, :, 1]
r3 = DF.c[:, :, 4]/DF.c[:, :, 1]
b = np.all(~np.isnan(DF.c[:, :, 1:]), axis=-1)
r1[b] = np.nan
r2[b] = np.nan
r3[b] = np.nan
s1 = np.nanmean(r1, axis=0)
s2 = np.nanmean(r2, axis=0)
s3 = np.nanmean(r3, axis=0)
b = count[:, k] > 50
axM = mapplot.mapPoint(
    figM, gsM[0, 0], latA[b], lonA[b], s1[b], s=10, cb=True)
axM.set_title('N-org / TN')
axM = mapplot.mapPoint(
    figM, gsM[1, 0], latA[b], lonA[b], s2[b], s=10, cb=True)
axM.set_title('NO3 / TN')
axM = mapplot.mapPoint(
    figM, gsM[2, 0], latA[b], lonA[b], s3[b], s=10, cb=True)
axM.set_title('NHx / TN')
figM.show()
