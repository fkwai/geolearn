"""
select site based on count of sample
bMat from app/data/usgsSample/index/createIndex.py
created in new index folder under dbUSGS

"""

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


saveFile = os.path.join(kPath.dirUsgs, 'index', 'bMat_A79_V20.npz')
npz = np.load(saveFile)
matC = npz['matC']
matF = npz['matCF']
matQ = npz['matQ']
tR = npz['tR']
codeLst = list(npz['codeLst'])
siteNoLst = list(npz['siteNoLst'])

matB = matC * ~matF * matQ[:, :, None]
matCount1 = np.sum(matB, axis=0)

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