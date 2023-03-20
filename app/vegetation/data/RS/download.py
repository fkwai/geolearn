import os, argparse
import pandas as pd
import ee
import time
from datetime import datetime, timedelta
from collections import OrderedDict
from calendar import monthrange
from hydroDL import kPath
from hydroDL.data.gee import product, geeutils
import json
import time

# load location
# crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
# tabCrd = pd.read_csv(crdFile, index_col=0)
outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_single.json')
with open(outFile, 'r') as fp:
    dictLst = json.load(fp)

ee.Initialize()

scale = 500
sd = '2015-01-01'
ed = '2023-03-01'

# col = product.sentinel1(sd, ed)
# outFolder = os.path.join(kPath.dirVeg, 'RS', 'sentinel1')
col = product.landset8(sd, ed)
outFolder = os.path.join(kPath.dirVeg, 'RS', 'landset')

if not os.path.exists(outFolder):
    os.mkdir(outFolder)


def download(col,outFolder):
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)
    # for ind, row in tabCrd.iterrows():
    for k, siteDict in enumerate(dictLst):
        t0 = time.time()
        lat = siteDict['crd'][0]
        lon = siteDict['crd'][1]
        siteId = siteDict['siteId']
        fileName = os.path.join(outFolder, siteId + '.csv')
        geometry = ee.Geometry.Point([lon, lat])
        region = col.getRegion(geometry, int(scale)).getInfo()
        df = geeutils.record2df(region)
        # df.drop(columns=['longitude','latitude'], inplace=True)
        df.to_csv(fileName, index=False)
        print(siteId, time.time() - t0)

col = product.sentinel1(sd, ed)
outFolder = os.path.join(kPath.dirVeg, 'RS', 'sentinel1')
download(col,outFolder)

col = product.landset8(sd, ed)
outFolder = os.path.join(kPath.dirVeg, 'RS', 'landset')
download(col,outFolder)

col = (
    ee.ImageCollection('LARSE/GEDI/GEDI02_B_002_MONTHLY').filterDate(sd, ed)
    .select('solar_elevation')
    # .filter(ee.Filter.eq('CLOUD_COVER', 0))
)
siteDict=dictLst[0]
lat = siteDict['crd'][0]
lon = siteDict['crd'][1]
siteId = siteDict['siteId']
fileName = os.path.join(outFolder, siteId + '.csv')
geometry = ee.Geometry.Point([lon, lat])
region = col.getRegion(geometry, int(1000)).getInfo()
df = geeutils.record2df(region)
