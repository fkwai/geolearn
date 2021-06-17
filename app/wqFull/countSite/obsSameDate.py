import importlib
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

"""
plot the rate when code i is observed when code j is observed
"""
# count - only for C now
saveFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'bMat.npz')
npz = np.load(saveFile)
matC = npz['matC']
matCF = npz['matCF']
matQ = npz['matQ']
tR = npz['tR']
codeLst = list(npz['codeLst'])
siteNoLst = list(npz['siteNoLst'])

# remove flags
matB = matC & (~matCF)
countB = np.sum(matB, axis=1)

# pick sites that have at least [the] observation days
# codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
# codeSel = ['00405','00600', '00605', '00618', '00660', '00665','71846']
codeSel = usgs.newC
indC = [codeLst.index(code) for code in codeSel]
mat = matB[:, :, indC]
the = 0
count = np.sum(np.any(mat, axis=2), axis=1)
indS = np.where(count > the)[0]
nc = len(codeSel)
out = np.ndarray([nc, nc])
for j, codej in enumerate(codeSel):
    cj = codeLst.index(codej)
    for i, codei in enumerate(codeSel):
        ci = codeLst.index(codei)
        if i == j:
            a = matB[indS, :, cj]
            b1 = np.any(matB[indS, :, :cj], axis=2)
            b2 = np.any(matB[indS, :, cj+1:], axis=2)
            b = b1 | b2
            # at least one other is observed
            out[j, i] = 1-np.sum(a & b)/np.sum(a)
        else:
            a = matB[indS, :, cj]
            b = matB[indS, :, ci]
            out[j, i] = np.sum(a & b)/np.sum(a)

labelLst = ['{} {}'.format(usgs.codePdf.loc[code]['shortName'], code)
            for code in codeSel]
fig, ax = plt.subplots(1, 1)
axplot.plotHeatMap(ax, out*100, labLst=labelLst)
fig.show()
