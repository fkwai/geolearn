import os
import time
import numpy as np
from hydroDL.data import gridMET
from hydroDL import kPath

# save all mask to one file
ncFile = os.path.join(kPath.dirData, 'gridMET', 'etr_1979.nc')
data = gridMET.readNcData(ncFile)
nanMaskAll = np.isnan(data)
nanMask = nanMaskAll.any(axis=0)


maskDir = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'mask')
siteNoLst = [f[:-4] for f in sorted(os.listdir(maskDir)) if f[-3:] == 'npy']
maskLst = list()
k = 0
t0 = time.time()
for siteNo in siteNoLst:
    mask = np.load(os.path.join(maskDir, siteNo+'.npy'))
    mask[nanMask] = 0
    mask = mask/np.sum(mask)
    maskLst.append(mask)
    print('{} {} time {}'.format(k, siteNo, time.time()-t0))
    k = k+1
maskAll = np.stack(maskLst, axis=2)
np.save(os.path.join(kPath.dirData, 'USGS', 'gridMET', 'maskAll.npy'), maskAll)

# get nan sites (too small no mask)
aa = np.sum(maskAll.reshape(-1, len(siteNoLst)), axis=0)
# for k in range(ns):
#     print(k,aa[k])
for kk in list(np.where(np.isnan(aa))[0]):
    print(siteNoLst[kk])
