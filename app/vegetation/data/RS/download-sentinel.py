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
crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd = pd.read_csv(crdFile, index_col=0)
# outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_single.json')
# with open(outFile, 'r') as fp:
#     dictLst = json.load(fp)

ee.Initialize()

scale = 500
sd = '2015-01-01'
ed = '2023-03-01'


def download(col, outFolder):
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)
    for ind, row in tabCrd.iterrows():
        # for k, siteDict in enumerate(dictLst):
        t0 = time.time()
        lat = row['lat']
        lon = row['lon']
        siteId = row.name
        fileName = os.path.join(outFolder, siteId + '.csv')
        if not os.path.exists(fileName):
            try:
                geometry = ee.Geometry.Point([lon, lat])
                region = col.getRegion(geometry, int(scale)).getInfo()
                df = geeutils.record2df(region)
                # df.drop(columns=['longitude','latitude'], inplace=True)
                df.to_csv(fileName, index=False)
                print(siteId, time.time() - t0)
            except:
                print('error', siteId, outFolder)


col = product.sentinel1(sd, ed)
outFolder = os.path.join(kPath.dirVeg, 'RS', 'sentinel1')
download(col, outFolder)

# col = product.landset8(sd, ed)
# outFolder = os.path.join(kPath.dirVeg, 'RS', 'landsat8')
# download(col, outFolder)
