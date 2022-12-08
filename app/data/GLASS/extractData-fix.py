
import time
from hydroDL import kPath
import os
import numpy as np
import pandas as pd
import json

maskDir = os.path.join(kPath.dirUSGS, 'GLASS', 'mask')
outDir = os.path.join(kPath.dirUSGS, 'GLASS', 'output')

# load data
# varLst = ['LAI', 'FAPAR', 'NPP']
varLst = ['LAI']
tempDir = os.path.join(kPath.dirRaw, 'GLASS', 'temp')
dictData = dict()
t0 = time.time()
for iV, var in enumerate(varLst):
    dataLst = list()
    for yr in np.arange(1982, 2019):
        t1 = time.time()
        tempFile = os.path.join(tempDir, '{}_{}.npz'.format(var, yr))
        dataTemp = np.load(tempFile)['out']
        dataLst.append(dataTemp)
        t2 = time.time()
        print('{} {} {:.2f}'.format(yr, var,  time.time()-t0), flush=True)
    dictData[var] = np.concatenate(dataLst, axis=2)
    print('{} {} {:.2f}'.format(yr, var, time.time()-t0))


# construct time
tLst, yrLst, dLst = [list(), list(), list()]
for yr in np.arange(1982, 2019):
    for d in np.arange(1, 366, 8):
        t = np.datetime64(str(yr))+np.timedelta64(d-1, 'D')
        tLst.append(t)
        yrLst.append(yr)
        dLst.append(d)
tAry = np.array(tLst)
nt = len(tAry)

# Select sites
fileSiteNo = os.path.join(kPath.dirUSGS, 'basins', 'siteNoLst.json')
with open(fileSiteNo) as fp:
    dictSite = json.load(fp)
siteNoLstAll = dictSite['CONUS']
siteNoLstTemp = [f for f in sorted(os.listdir(outDir))]
siteNoLst = [f for f in siteNoLstAll if f not in siteNoLstTemp]

# write to output
dictNan = {'LAI': 2550, 'FAPAR': 255, 'NPP': 65535}
for var in varLst:
    ind = dictData[var] == dictNan[var]
    dictData[var] = dictData[var].astype('float32')
    dictData[var][ind] = np.nan


[j1, j2] = [800, 1300]
[i1, i2] = [1100, 2300]

ns = len(siteNoLst)
maskLst = list()
t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    maskFile = os.path.join(maskDir, siteNo)
    mask = np.load(maskFile+'.npz')['mask'].astype('float32')
    temp = mask[j1:j2, i1:i2]
    maskLst.append(temp/np.sum(temp))
    print('{}/{} {:.2f}'.format(k, ns, time.time()-t0))
maskAry = np.stack(maskLst, axis=2)

# extract
data = dictData['LAI']
data[np.isnan(data)] = 0
m1 = data.reshape(-1, nt)
m2 = maskAry.reshape(-1, ns)
out = np.matmul(m1.T, m2)

for k, siteNo in enumerate(siteNoLst):
    df = pd.DataFrame(columns=varLst, data=out[:, k], index=tAry)
    df.index.name = 'date'
    df.to_csv(os.path.join(outDir, siteNo))
    print('saving {}'.format(siteNo))

# for k, siteNo in enumerate(siteNoLst):
#     t1 = time.time()
#     df = pd.DataFrame(columns=varLst, index=tAry)
#     for var in varLst:
#         data = dictData[var]
#         maskFile = os.path.join(maskDir, siteNo)
#         mask = np.load(maskFile+'.npz')['mask'][j1:j2, i1:i2].astype('float32')
#         out = np.nansum(data*mask[:, :, None], axis=(0, 1)) / \
#             np.nansum(mask, axis=(0, 1))
#         df[var] = out
#     df.index.name = 'date'
#     df.to_csv(os.path.join(outDir, siteNo))
#     print('{} {} {:.2f}'.format(k, siteNo, time.time()-t0))
