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


# remove flags
sd = np.datetime64('1982-01-01')
ed = np.datetime64('2018-12-31')
indT1 = np.where(tR == sd)[0][0]
indT2 = np.where(tR == ed)[0][0]
matB = matC & (~matCF)
matB = matB[:, indT1:indT2+1, :]
countB = np.sum(matB, axis=1)
countT = np.sum(np.any(matB, axis=2), axis=1)

# plot the vs nSite
a = np.sort(countT)[::-1]
b = np.arange(len(countT))+1
aa = np.sort(countB)[:, ::-1]

fig, ax = plt.subplots(1, 1)
for code in usgs.newC:
    ic = codeLst.index(code)
    aa = np.sort(countB[:, ic])[::-1]
    ax.plot(aa, np.log10(b), label=code)
ax.legend()
fig.show()

theAry = np.arange(100, 800)
out1 = np.ndarray(len(theAry))
out2 = np.ndarray(len(theAry))
for k, the in enumerate(theAry):
    varC = usgs.newC
    indC = [codeLst.index(code) for code in varC]
    pickMat = countB[:, indC] > the
    mat1 = np.sum(pickMat, axis=1)
    mat2 = np.sum(pickMat[:, 2:], axis=1)
    out1[k] = np.mean(mat1[mat1 > 0])
    out2[k] = np.mean(mat1[mat2 > 0])
    # out1[k] = np.sum(mat1 > 0)
    # out2[k] = np.sum(mat2 > 0)

fig, ax = plt.subplots(1, 1)
ax.plot(theAry, out1, 'r')
ax.plot(theAry, out2, 'b')
fig.show()

# selected sites with obs more than xxx
the = 200
varC = usgs.newC
# varC = ['00405','00600', '00605', '00618', '00660', '00665','71846']
# varC = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
indC = [codeLst.index(code) for code in varC]
pickMat = countB[:, indC] > the
dictSite = dict()
# each code
for ic, code in enumerate(varC):
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
        temp[:, varC.index(code)] = False
    indS = np.where(np.any(temp, axis=1))[0]
    dictSite[name] = [siteNoLst[ind] for ind in indS]

# print results
for key in dictSite.keys():
    print(key, len(dictSite[key]))

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
saveName = os.path.join(dirInv, 'siteSel', 'dictG{}'.format(the))
with open(saveName+'.json', 'w') as fp:
    json.dump(dictSite, fp, indent=4)
