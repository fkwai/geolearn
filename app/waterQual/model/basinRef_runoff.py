import sklearn.tree
from hydroDL.master import basins
from hydroDL.app import waterQuality, relaCQ
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('basinRef')


# calculate run-off
siteNoLst = wqData.info.siteNo.unique().tolist()
dfArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoLst)
dfArea.rename({'STAID': 'siteNo'})

info = wqData.info
area = info.join(dfArea, on='siteNo')['DRAIN_SQKM'].values
unitConv = 0.3048**3*365*24*60*60/1000**2

q = wqData.q[:, :, 0]
runoff = q/area*unitConv

i=np.random.randint(365)
j=np.random.randint(len(info))
q[i,j]
area[j]
runoff[i,j]
q[i,j]/area[j]


# nan in Q - 3% all nan, 7% any nan

len(np.where(np.isnan(q).all(axis=0))[0])
len(np.where(np.isnan(q).any(axis=0))[0])

iNan=np.where(np.isnan(q))
iRow, cRow = np.unique(iNan[1], return_counts=True)
len(np.where(cRow>300)[0])

# compare of opt1-4
outLst = ['basinRef-opt1', 'basinRef-opt2']
trainSet = 'first80'
testSet = 'last20'
pLst1, pLst2, errMatLst1, errMatLst2 = [list() for x in range(4)]
for outName in outLst:
    p1, o1 = basins.testModel(outName, trainSet, wqData=wqData, ep=200)
    p2, o2 = basins.testModel(outName, testSet, wqData=wqData, ep=200)
    errMat1 = wqData.errBySite(p1, subset=trainSet)
    errMat2 = wqData.errBySite(p2, subset=testSet)
    pLst1.append(p1)
    pLst2.append(p2)
    errMatLst1.append(errMat1)
    errMatLst2.append(errMat2)
