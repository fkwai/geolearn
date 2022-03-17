
from mpl_toolkits import basemap
import pandas as pd
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.app.waterQuality import WRTDS
import statsmodels.api as sm
import scipy
from hydroDL.app.waterQuality import cqType
import importlib
import time

# load data
dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)
siteNoLst = DF.siteNoLst
codeLst = DF.varC
ns = len(siteNoLst)
nc = len(codeLst)

# load pars
filePar = os.path.join(kPath.dirWQ, 'modelStat', 'typeCQ', dataName+'.npz')
npz = np.load(filePar)
matA = npz['matA']
matB = npz['matB']
matP = npz['matP']

# get types
importlib.reload(axplot)
importlib.reload(cqType)
tp = cqType.par2type(matB, matP)

codePlot = ['00010', '00300', '00955', '00935', '00400',  '00095', '00915', '00925', '00930',
            '00940', '00945', '00600', '00618', '00405', '71846', '00660', '00605', '00665', '00681', '80154']

# plot heatmap
out = np.ndarray([nc, nc])
for j, codej in enumerate(codePlot):
    cj = codeLst.index(codej)
    for i, codei in enumerate(codePlot):
        ci = codeLst.index(codei)

        a = tp[:, cj].copy()
        b = tp[:, ci].copy()
        b[b == -1] = -2
        out[j, i] = np.sum(a == b)/np.sum((a >= 0) & (b >= 0))


labelLst = ['{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
            for code in codePlot]
fig, ax = plt.subplots(1, 1)
axplot.plotHeatMap(ax, out*100, labLst=labelLst)
fig.show()

set(codeLst).difference(set(codePlot))
