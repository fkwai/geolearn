from hydroDL.data import gridMET
from hydroDL.utils import gis
import numpy as np
import shapefile
from shapely.geometry import shape
import os
import time
import argparse

""" script to run on ICME
screen
srun --exclusive --time 8:0:0 --pty bash
source activate pytorch
python /home/kuaifang/GitHUB/geolearn/app/waterQual/gridMetMask-job.py -S 1481 -E 1500
"""


def runJob(iStart, iEnd):
    workDir = r'/home/kuaifang/waterQuality/'
    ncFile = r'/home/kuaifang/Data/gridMET/pr_2010.nc'
    saveDir = r'/home/kuaifang/Data/USGS-mask/'
    t, lat, lon = gridMET.readNcInfo(ncFile)

    modelDir = os.path.join(workDir, 'modelUsgs2')
    sf = shapefile.Reader(os.path.join(modelDir, 'basinSel_prj.shp'))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', dest='iStart', type=int)
    parser.add_argument('-E', dest='iEnd', type=int)
    args = parser.parse_args()
    runJob(args.iStart, args.iEnd)
