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
import numpy as np
import hydroDL.data.nlcd

# load location
crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd = pd.read_csv(crdFile, index_col=0)
ee.Initialize()

sd = '2015-01-01'
ed = '2023-03-01'

outFolder = os.path.join(kPath.dirVeg, 'const')

nlcd2016 = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2016').select('landcover')
scale2016 = nlcd2016.projection().nominalScale().getInfo()
nlcd2019 = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2016').select('landcover')
scale2019 = nlcd2016.projection().nominalScale().getInfo()

latAll = tabCrd['lat'].values
lonAll = tabCrd['lon'].values
codeLst = ['nlcd2016','nlcd2019']
dfTemp = pd.DataFrame(index=tabCrd.index, columns=codeLst).fillna(0)
df = tabCrd[['lat', 'lon']].join(dfTemp)
t0 = time.time()
for k, siteId in enumerate(tabCrd.index.tolist()):
    # siteId=dfSite.index.tolist()[0]
    lat = tabCrd.loc[siteId]['lat']
    lon = tabCrd.loc[siteId]['lon']
    if np.isnan(lat) or np.isnan(lon):
        continue
    point = ee.Geometry.Point([lon, lat])
    t1 = time.time()
    for col, nlcd, scale in zip(['nlcd2016','nlcd2019'], [nlcd2016, nlcd2019], [scale2016, scale2019]):
        try:
            data = nlcd2019.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=scale).getInfo()
            df.at[siteId, col] = data['landcover']
            print('{} {} {:.2f} {:.2f}'.format(k, siteId, time.time() - t0, time.time() - t1))
        except Exception:
            print('error {}'.format(siteId))
df.to_csv(os.path.join(outFolder, 'nlcd-point.csv'))

