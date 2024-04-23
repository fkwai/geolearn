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
dfSite = pd.DataFrame(
    columns=['siteId', 'siteName', 'state', 'fuel', 'gacc', 'lat', 'lon']
)
for k, siteDict in enumerate(dictLst):
    dfSite.loc[k] = [
        siteDict['siteId'],
        siteDict['siteName'],
        siteDict['state'],
        siteDict['fuel'],
        siteDict['gacc'],
        siteDict['crd'][0],
        siteDict['crd'][1],
    ]
dfSite = dfSite.set_index('siteId').sort_index()

ee.Initialize()

sd = '2015-01-01'
ed = '2023-03-01'


def download(colStr, scale, dfSite):
    outFolder = os.path.join(kPath.dirVeg, 'RS', '{}-{}m'.format(colStr, scale))
    col = getattr(product, colStr)(sd, ed)
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)
    for ind, row in dfSite.iterrows():
        t0 = time.time()
        siteId = ind
        lat = row['lat']
        lon = row['lon']
        fileName = os.path.join(outFolder, siteId + '.csv')
        if not os.path.exists(fileName):            
            geometry = ee.Geometry.Point([lon, lat])
            region = col.getRegion(geometry, int(scale)).getInfo()
            df = geeutils.record2df(region)
            df.to_csv(fileName, index=False)
            print(colStr, scale, siteId, time.time() - t0)


productLst = ['MYD09GA', 'MOD09GA']
scale = 500
for p in productLst:
    download(p, scale, dfSite)
