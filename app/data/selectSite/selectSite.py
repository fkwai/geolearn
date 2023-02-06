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
siteNoLst = list(npz['siteNoLst'])

# selected sites with obs more than xxx
the = 200
varC = usgs.newC
countMat = np.sum(matC*~matCF, axis=1)
pickMat = countMat > the
dictSite = dict()
# each code
for code in codeLst:
    ic = codeLst.index(code)
    indS = np.where(pickMat[:, ic])[0]
    dictSite[code] = [siteNoLst[ind] for ind in indS]
# all
indS = np.where(np.any(pickMat, axis=1))[0]
dictSite['comb'] = [siteNoLst[ind] for ind in indS]
# rm some
rmLst = [['00010'], ['00010', '00095'], ['00010', '00095', '00400']]
nameLst = ['rmT', 'rmTK', 'rmTKH']
for rmCode, name in zip(rmLst, nameLst):
    temp = pickMat.copy()
    for code in rmCode:
        temp[:, codeLst.index(code)] = False
    indS = np.where(np.any(temp, axis=1))[0]
    dictSite[name] = [siteNoLst[ind] for ind in indS]

# print results
for key in dictSite.keys():
    print(key, len(dictSite[key]))
    