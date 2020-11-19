from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
import json


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
codeLst = sorted(usgs.newC)
ep = 500
reTest = True
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)

codeLst = sorted(usgs.newC)
info = wqData.info

out = np.ndarray([len(codeLst), len(codeLst)])
for k, code in enumerate(codeLst):
    ic = wqData.varC.index(code)
    siteNoCode = dictSite[code]
    bs = info['siteNo'].isin(siteNoCode)
    bv = ~np.isnan(wqData.c[:, wqData.varC.index(code)])
    ind = info.index[bs & bv].values
    mat = wqData.c[ind, :]
    count = np.sum(~np.isnan(mat), axis=0)
    n = count[ic]
    countP = count/n
    for j, code2 in enumerate(codeLst):
        ic2 = wqData.varC.index(code2)
        out[k, j] = countP[ic2]

fig, ax = plt.subplots(1, 1)
axplot.plotHeatMap(ax, out*100, codeLst)
fig.show()
