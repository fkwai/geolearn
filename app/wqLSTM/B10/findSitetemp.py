import importlib
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

# constrain date
sd = np.datetime64('1982-01-01')
ed = np.datetime64('2018-12-31')
indT1 = np.where(tR == sd)[0][0]
indT2 = np.where(tR == ed)[0][0]
indC = np.array([codeLst.index(x) for x in usgs.varC])

dictSite = dict()
dictFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'dictSite.json')

# G150B10 - B10>150, A10 >50 with any obs
subName = 'N150B10'
tB = np.datetime64('2010-01-01')
indTB = np.where(tR == tB)[0][0]
matB1 = matB[:, indT1:indTB, indC]
matB2 = matB[:, indTB:indT2, indC]
count1 = np.sum(np.any(matB1, axis=-1), axis=-1)
count2 = np.sum(np.any(matB2, axis=-1), axis=-1)
ind = np.where((count1 >= 150) & (count2 >= 50))[0]
siteNoSel = [siteNoLst[x] for x in ind]
dictSite[subName] = siteNoSel
with open(dictFile, 'w') as fp:
    json.dump(dictSite, fp, indent=4)

# G150B10 - B10>150, A10 >50 with single variable
subName = 'N150B10S'
count1 = np.sum(matB1, axis=1)
count2 = np.sum(matB2, axis=1)
ind = np.where(np.any(count1 >= 150, axis=1) &
               np.any(count2 >= 50, axis=1))[0]
siteNoSel = [siteNoLst[x] for x in ind]
dictSite[subName] = siteNoSel
with open(dictFile, 'w') as fp:
    json.dump(dictSite, fp, indent=4)

len(dictSite['N150B10'])

len(dictSite['N150B10S'])

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y28N5.json')) as f:
    dictSite = json.load(f)
