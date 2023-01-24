import numpy as np
import shapefile
from shapely.geometry import shape
import os
import time
import argparse

from hydroDL.data import gridMET
from hydroDL.utils import gis
from hydroDL import kPath

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', dest='iStart', type=int, default=7000)
    parser.add_argument('-E', dest='iEnd', type=int, default=7111)
    parser.add_argument('-R', dest='reMask', type=int, default=True)
    args = parser.parse_args()
    iStart = args.iStart
    iEnd = args.iEnd
    reMask = args.reMask


ncFile = os.path.join(kPath.dirData, 'gridMET', 'etr_1979.nc')
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basinAll_prj.shp')
saveDir = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'mask')

t, lat, lon = gridMET.readNcInfo(ncFile)
sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]

if reMask is False:
    maskLst = [f[:-4] for f in os.listdir(saveDir) if f[-3:] == 'npy']
    tempShpLst = list()
    tempNoLst = list()
    for shp, siteNo in zip(shapeLst, siteNoLst):
        if siteNo not in maskLst:
            tempShpLst.append(shp)
            tempNoLst.append(siteNo)
    shapeLst = tempShpLst
    siteNoLst = tempNoLst

if iEnd == 0:  # do mask for every basin
    iEnd = len(siteNoLst)
    iStart = 0

t0 = time.time()
for k in range(iStart, iEnd):
    t1 = time.time()
    geog = shape(shapeLst[k])
    mask = gis.gridMask(lat, lon, geog)
    print('basin {} {:.2f}'.format(
        k, time.time()-t1))
    outFile = os.path.join(saveDir, siteNoLst[k])
    np.save(outFile, mask)
print('total time {}'.format(time.time() - t0))


# """ script to run on ICME
# screen
# srun --exclusive --time 8:0:0 --pty bash
# source activate pytorch
# python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetMask-job.py -S 1481 -E 1500
# """

# """ script to run on sherlock
# app/waterQual/data/slurmScript.py
# """
