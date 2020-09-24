from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import scipy
import json

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
countMatD = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMatW = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))


ny = 3
ns = 36
indLst = list()
for code in codeLst:
    ic = codeLst.index(code)
    count = np.sum(countMatW[:, -ny:, ic], axis=1)
    indS = np.where(count >= ns)[0]
    indLst.append(indS)
    print('{} {}'.format(code, len(indS)))
# get rid of 00010 and 00095
indAll = np.unique(np.concatenate(indLst[2:]))
print(len(indAll))
indAll = np.unique(np.concatenate(indLst[1:]))
print(len(indAll))
indAll = np.unique(np.concatenate(indLst))
print(len(indAll))

# save to dict
dictSite = dict()
for k, code in enumerate(codeLst):
    siteNoLst = [siteNoLstAll[ind] for ind in indLst[k]]
    dictSite[code] = siteNoLst
indComb = np.unique(np.concatenate(indLst))
dictSite['comb'] = [siteNoLstAll[ind] for ind in indComb]
saveName = os.path.join(dirInv, 'siteSel', 'dictNB_y16n36')
with open(saveName+'.json', 'w') as fp:
    json.dump(dictSite, fp, indent=4)
