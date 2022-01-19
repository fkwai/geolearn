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


# devide train / test
usgs.codePdf['shortName']
codeSel = ['00915', '00940', '00945', '00955']
t1 = np.datetime64('1982-01-01', 'D')
t2 = np.datetime64('2009-10-01', 'D')
t3 = np.datetime64('2019-01-01', 'D')
indT1 = np.where((tR >= t1) & (tR < t2))[0]
indT2 = np.where((tR >= t2) & (tR < t3))[0]
indC = [codeLst.index(code) for code in codeSel]

mat = matB[:, :, indC]
mat1 = matB[:, indT1, :][:, :, indC]
mat2 = matB[:, indT2, :][:, :, indC]
count = np.sum(np.any(mat, axis=2), axis=1)
count1 = np.sum(np.any(mat1, axis=2), axis=1)
count2 = np.sum(np.any(mat2, axis=2), axis=1)

# pick
pickMat = (count1 >= 150) & (count2 > 50)
len(np.where(pickMat)[0])
indS = np.where(pickMat)[0]
siteNoSel = [siteNoLst[ind] for ind in indS]

dataName = 'B5Y09a'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoSel, varC=codeSel, varG=gageII.varLstEx)
DF = dbBasin.DataFrameBasin(dataName)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')

# each variable
for k, code in enumerate(codeSel):
    c1 = np.sum(mat1[:, :, k], axis=1)
    c2 = np.sum(mat2[:, :, k], axis=1)
    pickMat = (c1 >= 150) & (c2 > 50)
    len(np.where(pickMat)[0])
    indS = np.where(pickMat)[0]
    siteNoCode = [siteNoLst[ind] for ind in indS]
    DF.saveSubset('WYB09-'+code, sd='1982-01-01', ed='2009-10-01')
    DF.saveSubset('WYA09-'+code, sd='2009-10-01', ed='2018-12-31')
