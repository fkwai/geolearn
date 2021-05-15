from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

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

# devide train / test
t1 = np.datetime64('1982-01-01', 'D')
t2 = np.datetime64('2010-01-01', 'D')
t3 = np.datetime64('2019-01-01', 'D')
indT1 = np.where((tR >= t1) & (tR < t2))[0]
indT2 = np.where((tR >= t2) & (tR < t3))[0]
indC = [codeLst.index(code) for code in usgs.newC]
mat1 = np.sum(matB[:, indT1, :][:, :, indC], axis=1)
mat2 = np.sum(matB[:, indT2, :][:, :, indC], axis=1)

# threshold
ns = 5
yr = tR.astype('M8[Y]')
ny1 = np.unique(yr[indT1]).shape[0]
ny2 = np.unique(yr[indT2]).shape[0]
th1 = ns*ny1
th2 = ns*ny2

# pick
pickMat = (mat1 >= th1) & (mat2 >= th2)
codePick = usgs.newC
dictSite = dict()
indS = np.where(np.any(pickMat, axis=1))[0]
dictSite['comb'] = [siteNoLst[ind] for ind in indS]
rmLst = [['00010'], ['00010', '00095'], ['00010', '00095', '00400']]
nameLst = ['rmT', 'rmTK', 'rmTKH']
for rmCode, name in zip(rmLst, nameLst):
    temp = pickMat.copy()
    for code in rmCode:
        temp[:, codePick.index(code)] = False
    indS = np.where(np.any(temp, axis=1))[0]
    dictSite[name] = [siteNoLst[ind] for ind in indS]
for code in codePick:
    ic = codePick.index(code)
    indS = np.where(pickMat[:, ic])[0]
    dictSite[code] = [siteNoLst[ind] for ind in indS]
for code in dictSite.keys():
    print(code, len(dictSite[code]))

# save
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
saveName = os.path.join(dirInv, 'siteSel', 'dictRB_Y28N{}'.format(ns))
with open(saveName+'.json', 'w') as fp:
    json.dump(dictSite, fp, indent=4)
