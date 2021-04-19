from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import json

saveFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'bMat.npz')
npz = np.load(saveFile)
matC = npz['matC']
matCF = npz['matCF']
matQ = npz['matQ']
tR = npz['tR']
codeLst = list(npz['codeLst'])
siteNoLst = npz['siteNoLst']

varC = usgs.newC
# selected sites with obs more than xxx
countMat = np.sum(matC*~matCF, axis=1)

tempC = ['00300', '00400', '00405', '00600', '00605',
         '00618', '00660', '00665', '00681', '00915',
         '00925', '00930', '00935', '00940', '00945',
         '00955', '71846', '80154']


nc = len(codeLst)
mat1 = np.zeros([nc, nc])
mat2 = np.zeros([nc, nc])
for j, c1 in enumerate(codeLst):
    a = matC[:, :, codeLst.index(c1)]
    for i, c2 in enumerate(codeLst):
        print(j, i)
        b = matC[:, :, codeLst.index(c2)]
        mat1[j, i] = np.sum(a*b)/np.sum(a)
        the = 200
        ix = np.sum(a, axis=1) > the
        mat2[j, i] = np.sum(a[ix, :]*b[ix, :])/np.sum(a[ix, :])

fig, ax = plt.subplots(1, 1)
axplot.plotHeatMap(ax, mat1*100, labLst=codeLst)
fig.show()

fig, ax = plt.subplots(1, 1)
axplot.plotHeatMap(ax, mat2*100, labLst=codeLst)
fig.show()
