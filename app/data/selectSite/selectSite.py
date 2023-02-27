from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
import json

saveFile = os.path.join(kPath.dirUsgs, 'siteSel', 'matBool.npz')
npz = np.load(saveFile)
matC = npz['matC']
matF = npz['matF']
matQ = npz['matQ']
t = npz['t']
codeLst = list(npz['codeLst'])
siteNoLst = list(npz['siteNoLst'])

matB = matC * ~matF * matQ[:, :, None]
matCount = np.sum(matB, axis=0)

# threshold
the = 100
matPick = matCount[:, 2:] > the
dictSite = dict()

# all
indS = np.where(np.any(matPick, axis=1))[0]
name = 'any-B{}'.format(the)
dictSite[name] = [siteNoLst[ind] for ind in indS]
print(name, len(indS))

# each code
for code in codeLst:
    ic = codeLst.index(code)
    indS = np.where(matPick[:, ic])[0]
    name='{}-B{}'.format(code,the)
    dictSite[name] = [siteNoLst[ind] for ind in indS]
    print(code, len(indS))

# remove some variables
rmLst = [['00010'], ['00010', '00095'], ['00010', '00095', '00400']]
nameLst = ['rmT', 'rmTK', 'rmTKH']
for rmCode, nameStr in zip(rmLst, nameLst):
    temp = matPick.copy()
    for code in rmCode:
        temp[:, codeLst.index(code)] = False
    indS = np.where(np.any(temp, axis=1))[0]
    name = '{}-B{}'.format(nameStr, the)
    dictSite[name] = [siteNoLst[ind] for ind in indS]
    print(name, len(indS))

saveName = 'siteNoLst_79_23'
saveFile = os.path.join(kPath.dirUsgs, 'siteSel', saveName)
if os.path.exists(saveFile):
    with open(saveFile, 'r') as f:
        dictSiteOld = json.load(f)
    dictSite.update(dictSiteOld)
with open(saveFile, 'w') as f:
    json.dump(dictSite, f, indent=2)

# saveName = 'siteNoLst_79_23'
# saveFile = os.path.join(kPath.dirUsgs, 'siteSel', saveName)
# with open(saveFile, 'w') as f:
#     json.dump(dict(ny5_rmTK=siteNoLst), f, indent=2)

# dataName = 'NY5'
# DF = dbBasin.DataFrameBasin.new(
#     dataName,
#     siteNoLst,
#     varC=usgs.varC + ['00060', '00065'],
#     varF=gridMET.varLst,
#     varG=gageII.varLstEx,
#     edStr='2023-01-01',
# )
