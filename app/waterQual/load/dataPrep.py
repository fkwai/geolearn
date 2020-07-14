from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('basinRef')
# wqData.c = wqData.c * wqData.q[-1, :, 0:1]
# saveName = os.path.join(kPath.dirWQ, 'trainData','loadRef')
# np.savez(saveName, q=wqData.q, f=wqData.f,
#          c=wqData.c, g=wqData.g, cf=wqData.cf)

siteNoLst = wqData.siteNoLst
info = wqData.info
siteNoLst = wqData.siteNoLst
nc = 24
ut = np.full([len(info), nc], np.nan)
ul = np.full([len(info), nc], np.nan)

for siteNo in siteNoLst:
    indS = info[info['siteNo'] == siteNo].index.values
    for k in range(nc):
        v = wqData.c[indS, k]
        ut[indS, k] = np.nanpercentile(v, 95)
        ul[indS, k] = np.nanpercentile(v, 5)


c = (wqData.c-ul)/(ut-ul)
saveName = os.path.join(kPath.dirWQ, 'trainData', 'stanRef')
np.savez(saveName, q=wqData.q, f=wqData.f,
         c=c, g=wqData.g, ut=ut, ul=ul)
