import importlib
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import itertools

# FAILED ATTEMPT  - too slow

# count - only for C now
saveFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'bMat.npz')
npz = np.load(saveFile)
matC = npz['matC']
matCF = npz['matCF']
matQ = npz['matQ']
tR = npz['tR']
codeLst = list(npz['codeLst'])
siteNoLst = list(npz['siteNoLst'])
matB = matC & (~matCF)


# select subset
codeSel = usgs.varC
t1 = np.datetime64('1982-01-01', 'D')
t2 = np.datetime64('2009-10-01', 'D')
t3 = np.datetime64('2019-01-01', 'D')
indT1 = np.where((tR >= t1) & (tR < t2))[0]
indT2 = np.where((tR >= t2) & (tR < t3))[0]
indC = [codeLst.index(code) for code in codeSel]
matB1 = matB[:, indT1, :][:, :, indC]
matB2 = matB[:, indT2, :][:, :, indC]

subsetLst = list()
outLst = list()
combLst = list(itertools.combinations(usgs.varC, 3))
for kk, subset in enumerate(combLst):
    print(kk, len(combLst))
    indComb = [codeSel.index(s) for s in subset]
    temp = matB1[:, :, indComb]
    count1 = np.sum(matB1[:, :, indComb].all(axis=-1), axis=-1)
    count2 = np.sum(matB2[:, :, indComb].all(axis=-1), axis=-1)
    k = 5
    out = np.sum((count1 > 30*k) & (count2 > 10*k))
    subsetLst.append(subset)
    outLst.append(out)

count = np.sum(bTemp, axis=-1)
np.sum(count > 200)

a = np.sum(matB1[:, :, indComb].all(axis=-1), axis=-1)
