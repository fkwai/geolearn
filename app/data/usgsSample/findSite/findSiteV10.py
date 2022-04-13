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
codeSel = ['00405', '00600', '00605', '00660', '00665',
           '00915', '00925', '00930', '00935', '00945']
t1 = np.datetime64('1982-01-01', 'D')
t2 = np.datetime64('2009-10-01', 'D')
t3 = np.datetime64('2019-01-01', 'D')
indT1 = np.where((tR >= t1) & (tR < t2))[0]
indT2 = np.where((tR >= t2) & (tR < t3))[0]
indC = [codeLst.index(code) for code in codeSel]
matB1 = matB[:, indT1, :][:, :, indC]
matB2 = matB[:, indT2, :][:, :, indC]
matQ1 = matQ[:, indT1]
matQ2 = matQ[:, indT2]

# threshold: Q>90%; C-train>150; C-test>50;
