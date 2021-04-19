
from hydroDL import kPath
import os
import time
import shapefile
from shapely.geometry import shape
from hydroDL.utils import gis
import numpy as np
from joblib import Parallel, delayed

# read shapefiles
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basinAll_prj.shp')
sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]

# GLASS crd
saveDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'mask')
lon = np.arange(-179.975, 180, 0.05)
lat = np.arange(89.975, -90, -0.05)

t0 = time.time()


def func(siteNo):
    print('entering basin {}'.format(siteNo))
    t1 = time.time()
    k = siteNoLst.index(siteNo)
    geog = shape(shapeLst[k])
    mask = gis.gridMask(lat, lon, geog)
    outFile = os.path.join(saveDir, siteNoLst[k])
    np.savez_compressed(outFile, mask=mask)
    t2 = time.time()
    print('basin {}/{} {:.2f} {:.2f}'.format(k, len(siteNoLst), t2-t1, t2-t0))


results = Parallel(n_jobs=-1)(delayed(func)(siteNo) for siteNo in siteNoLst)
# seems that memory limited the parallel jobs - better run on