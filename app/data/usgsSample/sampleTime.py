import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib
import hydroDL.data.usgs.read as read


dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)

siteNoLst = DF.siteNoLst

# indtLst = list()
# tnLst = list()
# for k in range(len(siteNoLst)):
#     print(k)
#     siteNo = siteNoLst[k]
#     fileC = os.path.join(kPath.dirRaw, 'USGS', 'sample', siteNo)
#     dfC = read.readUsgsText(fileC, dataType='sample')
#     v = DF.c[:, k, :]
#     indVT = np.where((~np.isnan(v)).sum(axis=1) > 0)[0]
#     d1 = DF.t[indVT]
#     d2 = dfC['date'].values.astype('datetime64[D]')
#     d, ind1, ind2 = np.intersect1d(d1, d2, return_indices=True)
#     tm = dfC['sample_tm'].values[ind2]
#     tn = [t.hour + t.minute / 60 for t in tm]
#     indtLst.append(ind1)
#     tnLst.append(tn)
# matTM = np.full([len(DF.t), len(siteNoLst)], np.nan)
# for k in range(len(siteNoLst)):
#     matTM[indtLst[k], k] = tnLst[k]
# np.save(os.path.join(kPath.dirData, 'USGS', 'inventory', 'matTM.npy'), matTM)


# # fix bug
# mat = np.full([len(DF.t), len(DF.siteNoLst)], np.nan)
# matTM = np.load(os.path.join(kPath.dirData, 'USGS', 'inventory', 'matTM.npy'))
# for indS in range(matTM.shape[1]):
#     v = DF.c[:, indS, :]
#     ind1 = np.where((~np.isnan(v)).sum(axis=1) > 0)[0]
#     ind2=np.where(~np.isnan(matTM[:, indS]))[0]
#     mat[ind1[ind2], indS] = matTM[ind2, indS]
# matTM=mat
# np.save(os.path.join(kPath.dirData, 'USGS', 'inventory', 'matTM.npy'), matTM)


## 
matTM = np.load(os.path.join(kPath.dirData, 'USGS', 'inventory', 'matTM.npy'))

np.sum(~np.isnan(mat),axis=0)
# remove tm
bins = np.arange(0, 24.5, 0.5)
countMat = np.full([len(bins) - 1, 3], np.nan)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for k, code in enumerate(['00010', '00300', '00400']):
    indC = DF.varC.index(code)
    matT = matTM.copy()
    matT[~np.isnan(matT)]
    matT[np.isnan(DF.c[:, :, indC])] = np.nan
    for indS in range(matTM.shape[1]):        
        count = np.sum(~np.isnan(matT[:, indS]))
        # count2 = np.sum(~np.isnan(DF.c[:, indS,indC]))
        if count < 200:
            matT[:, indS] = np.nan
    out = axes[k].hist(matT[~np.isnan(matT)], bins=bins)
    axes[k].set_title(usgs.codePdf.loc[code]['shortName'])
    axes[k].set_xlabel('hour')
    if k==0:
        axes[k].set_ylabel('count')
    countMat[:, k] = out[0]
    k
fig.show()

# save countMat to csv
df = pd.DataFrame(countMat, columns=['00010', '00300', '00400'])


# find  missing
a=0
b=0
for k, code in enumerate(DF.varC):
    matT = matTM.copy()
    matC=DF.c[:, :, k]
    matT[np.isnan(matC)] = np.nan
    mat1=~np.isnan(matT)
    mat2=~np.isnan(matC)
    a=a+mat1.sum() 
    b=b+mat2.sum()
    print(code,mat1.sum()/mat2.sum())
a/b
dfC['sample_tm'].values