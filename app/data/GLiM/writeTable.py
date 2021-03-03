import os
from hydroDL import kPath
import numpy as np
import pandas as pd
import json
from osgeo import gdal
from hydroDL.data import GLiM
import time

maskDir = os.path.join(kPath.dirData, 'USGS', 'GLiM', 'mask_gageII_1KM')
rasterTiff = os.path.join(kPath.dirData, 'GLiM', 'NA_gageII_1KM.tif')

# siteNo
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

# read raster
tabCode = GLiM.tabCode
codeLst = tabCode['code'].tolist()
ds = gdal.Open(rasterTiff)
raster = ds.GetRasterBand(1).ReadAsArray()

# output
tabOut = pd.DataFrame(index=siteNoLst, columns=codeLst)
t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    mask = np.load(os.path.join(maskDir, siteNo+'.npy'))
    iy, ix = np.where(mask > 0)
    tempMask = mask[iy, ix]
    tempCode = raster[iy, ix]
    area = np.sum(tempMask)
    for code in codeLst:
        areaCode = np.sum(tempMask[tempCode == code])
        tabOut.at[siteNo, code] = areaCode/area
    print('{} {} {:.2f}'.format(k, siteNo, time.time()-t0))
tabOut.index.name = 'siteNo'
tabOut.to_csv(os.path.join(kPath.dirData, 'USGS', 'GLiM', 'tab_1KM'))
