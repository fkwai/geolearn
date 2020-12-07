import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality, relaCQ
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot

from hydroDL import utils
import numpy as np
import matplotlib.pyplot as plt
import time


wqData = waterQuality.DataModelWQ('basinRef')
siteNoLst = wqData.info.siteNo.unique().tolist()

c = wqData.c
q = wqData.q[np.arange(4, 365, 15), :, 0]
varC = wqData.varC
varNameLst = usgs.codePdf.loc[varC]['shortName'].tolist()
qNameLst = list(reversed(['{}d'.format(x) for x in range(0, 365, 15)]))
nc = c.shape[1]
nq = q.shape[0]


# calculate all at once
matCorr = np.full([nc, nq], np.nan)
for j in range(nc):
    for i in range(nq):
        (a, b), kk = utils.rmNan([c[:, j], q[i, :]])
        if len(kk) > 0:
            matCorr[j, i] = np.corrcoef(a, b)[0, 1]

importlib.reload(axplot)
fig, ax = plt.subplots()
axplot.plotHeatMap(ax, matCorr*100, [varNameLst, qNameLst])
fig.tight_layout()
fig.show()


# calculate site by site
ns = len(siteNoLst)
matCorrAll = np.full([nc, nc, ns], np.nan)
for k in range(ns):
    siteNo = siteNoLst[k]
    ind = wqData.info[wqData.info['siteNo'] == siteNo].index
    c = wqData.c[ind]
    for j in range(nc):
        for i in range(nc):
            (a, b), kk = utils.rmNan([c[:, j], c[:, i]])
            if len(kk) > 0:
                matCorrAll[j, i, k] = np.corrcoef(a, b)[0, 1]

varG = ['DRAIN_SQKM', 'ECO2_BAS_DOM', 'NUTR_BAS_DOM',
        'HLR_BAS_DOM_100M']
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)

name = 'ECO2_BAS_DOM'
# vLst = np.sort(tabG[name].unique()).tolist()
vLst = [8.2, 11.1]
titleLst = ['CENTRAL USA PLAINS', 'MEDITERRANEAN CALIFORNIA']
fig, axes = plt.subplots(1, 2)
for k in range(2):
    siteNo = tabG[tabG[name] == vLst[k]].index.tolist()
    ind = [siteNoLst.index(s) for s in siteNo]
    axplot.plotHeatMap(axes[k], np.nanmean(
        matCorrAll[:, :, ind], axis=2)*100, varNameLst)
    axes[k].set_title(titleLst[k])
fig.tight_layout()
fig.show()
