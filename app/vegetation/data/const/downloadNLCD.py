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

productM = 'MODIS/061/MCD43A4'
colM = ee.ImageCollection(productM).filterDate(sd, ed)
# optLst = ['nadgrid', 'modisgrid']
optLst = ['nadgrid']
outFolder = os.path.join(kPath.dirVeg, 'const')

nlcd2016 = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2016').select('landcover')
scale2016 = nlcd2016.projection().nominalScale().getInfo()
nlcd2019 = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2016').select('landcover')
scale2019 = nlcd2016.projection().nominalScale().getInfo()
for opt in optLst:
    if opt == 'nadgrid':
        proj = ee.Projection('EPSG:5072').getInfo()
    elif opt == 'modisgrid':
        proj = colM.first().projection().getInfo()
        dxM = proj['transform'][0]
        dyM = proj['transform'][4]
    errLst=list()
    latAll = tabCrd['lat'].values
    lonAll = tabCrd['lon'].values
    codeLst = list(hydroDL.data.nlcd.dictCode.keys())
    dfTemp = pd.DataFrame(index=tabCrd.index, columns=codeLst).fillna(0)
    df = tabCrd[['lat', 'lon']].join(dfTemp)
    df2016 = df.copy()
    df2019 = df.copy()
    t0 = time.time()
    for k, siteId in enumerate(tabCrd.index.tolist()):
        # siteId=dfSite.index.tolist()[0]
        lat = tabCrd.loc[siteId]['lat']
        lon = tabCrd.loc[siteId]['lon']
        if np.isnan(lat) or np.isnan(lon):
            continue
        point = ee.Geometry.Point([lon, lat])
        # create BB
        if opt == 'nadgrid':
            pointM = point.transform(proj['crs'])
            x, y = pointM.getInfo()['coordinates']
            x1 = x - 250
            y1 = y - 250
            x2 = x1 + 500
            y2 = y1 + 500
        elif opt == 'modisgrid':
            pointM = point.transform(proj['crs'])
            x, y = pointM.getInfo()['coordinates']
            xmod = np.mod(x, dxM)
            ymod = np.mod(y, dyM)
            x1 = x - xmod
            y2 = y - ymod
            x2 = x1 + dxM
            y1 = y2 + dyM
        if not (x > x1 and x < x2 and y > y1 and y < y2):
            raise Exception('point not in bb')
        bb = ee.Geometry.Rectangle([x1, y1, x2, y2], proj['crs'], False)
        t1 = time.time()
        for df, nlcd, scale in zip([df2016, df2019], [nlcd2016, nlcd2019], [scale2016, scale2019]):
            try:
                data = nlcd2019.reduceRegion(reducer=ee.Reducer.toList(), geometry=bb, scale=scale).getInfo()
                values, counts = np.unique(data['landcover'], return_counts=True)
                for v, c in zip(values, counts):
                    df.at[siteId, v] = c
                print('{} {} {:.2f} {:.2f}'.format(k, siteId, time.time() - t0, time.time() - t1))
            except Exception:
                print('error {}'.format(siteId))
                errLst.append(siteId)
    df2016.to_csv(os.path.join(outFolder, 'nlcd2016-{}.csv'.format(opt)))
    df2019.to_csv(os.path.join(outFolder, 'nlcd2019-{}.csv'.format(opt)))
