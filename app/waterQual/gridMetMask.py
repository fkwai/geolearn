from hydroDL.post.draw import map
import importlib
from hydroDL.data import gridMET
from hydroDL.utils import gis
import numpy as np
import shapefile
from shapely.geometry import shape
import os
import time
workDir = r'/home/kuaifang/waterQuality/'
ncFile = r'/home/kuaifang/Data/gridMET/pr_2010.nc'
t, lat, lon = gridMET.readNcInfo(ncFile)

modelDir = os.path.join(workDir, 'modelUsgs2')
sf = shapefile.Reader(os.path.join(modelDir, 'basinSel_prj.shp'))
shapeLst = sf.shapes()


nBasin = len(shapeLst)
maskAll = np.ndarray([nBasin, len(lat), len(lon)])
t0 = time.time()
for k in range(nBasin):
    t1 = time.time()
    geog = shape(shapeLst[k])
    mask = gis.gridMask(lat, lon, geog,ns=2)
    maskAll[k, :, :] = mask
    print('basin {} {:.2f}% {:.2f}s'.format(
        k, k / nBasin * 100, time.time()-t1))
print('total time {}'.format(time.time() - t0))

outFile = os.path.join(modelDir, 'basinMask_gridMet')
np.save(outFile, maskAll)
