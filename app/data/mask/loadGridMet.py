from iris.analysis import geometry
import iris
from iris.cube import Cube
from iris.coords import DimCoord
import numpy as np
import shapefile
from shapely.geometry import shape
import os
import time
import argparse
import matplotlib.pyplot as plt
import hydroDL.data.gridMET.io as io
from hydroDL.utils import gis
from hydroDL import kPath
import time


import json
fileSiteNo = os.path.join(kPath.dirData, 'USGS',
                          'basins', 'siteNoLst.json')
with open(fileSiteNo) as fp:
    dictSite = json.load(fp)
siteNoLst = dictSite['CONUS']

maskFolder = os.path.join(kPath.dirData, 'USGS', 'mask', 'gridMET')

t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    outFile = os.path.join(maskFolder, siteNoLst[k]+'.npz')
    # t1 = time.time()
    mask = np.load(outFile)
    # t2 = time.time()
    # print('basin {} {:.2f} {:.2f}'.format(
    #     k, t2-t1, t2-t0))
