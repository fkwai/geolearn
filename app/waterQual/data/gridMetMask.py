import numpy as np
import shapefile
from shapely.geometry import shape
import os
import time
import argparse

from hydroDL.data import gridMET
from hydroDL.utils import gis
from hydroDL import kPath

""" script to run on ICME
screen
srun --exclusive --time 8:0:0 --pty bash
source activate pytorch
python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetMask-job.py -S 1481 -E 1500
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', dest='iStart', type=int, default=0)
    parser.add_argument('-E', dest='iEnd', type=int, default=10)
    args = parser.parse_args()
    iStart = args.iStart
    iEnd = args.iEnd


ncFile = os.path.join(kPath.dirData, 'gridMET', 'etr_1979.nc')
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basinAll_prj.shp')
saveDir = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'mask')

t, lat, lon = gridMET.readNcInfo(ncFile)
sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]

t0 = time.time()
for k in range(iStart, iEnd):
    t1 = time.time()
    geog = shape(shapeLst[k])
    mask = gis.gridMask(lat, lon, geog, ns=4)
    print('basin {} {:.2f}'.format(
        k, time.time()-t1))
    outFile = os.path.join(saveDir, siteNoLst[k])
    np.save(outFile, mask)
print('total time {}'.format(time.time() - t0))
