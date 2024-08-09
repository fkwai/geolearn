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
from datetime import datetime


# load location
crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd = pd.read_csv(crdFile, index_col=0)
ee.Initialize()

sd = '2015-01-01'
ed = '2023-03-01'

productM = 'MODIS/061/MCD43A4'
colM = ee.ImageCollection(productM).filterDate(sd, ed)
# optLst = ['nadgrid', 'modisgrid']
optLst = ['modisgrid']
outFolder = os.path.join(kPath.dirVeg, 'const')
dataset = ee.Image('USGS/3DEP/10m')
elevation = dataset.select('elevation')
slope = ee.Terrain.slope(elevation)
scale = dataset.projection().nominalScale().getInfo()
codeLst = ['elevMean', 'slopeMean', 'elevPoint', 'slopePoint']

for opt in optLst:
    if opt == 'nadgrid':
        proj = ee.Projection('EPSG:5072').getInfo()
    elif opt == 'modisgrid':
        proj = colM.first().projection().getInfo()
        dxM = proj['transform'][0]
        dyM = proj['transform'][4]
    errLst = list()
    latAll = tabCrd['lat'].values
    lonAll = tabCrd['lon'].values
    outFile = os.path.join(outFolder, 'elev-{}.csv'.format(opt))
    if os.path.exists(outFile):
        df = pd.read_csv(outFile, index_col=0)
        siteResume = df[codeLst].last_valid_index()
        kResume = tabCrd.index.tolist().index(siteResume)
    else:
        dfTemp = pd.DataFrame(index=tabCrd.index, columns=codeLst)
        df = tabCrd[['lat', 'lon']].join(dfTemp)
        kResume = 0
    t0 = time.time()
    for k, siteId in enumerate(tabCrd.index.tolist()[kResume:]):
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
        elevM = elevation.reduceRegion(reducer=ee.Reducer.mean(), geometry=bb, scale=scale).getInfo()
        slopeM = slope.reduceRegion(reducer=ee.Reducer.mean(), geometry=bb, scale=scale).getInfo()
        elevP = elevation.reduceRegion(reducer=ee.Reducer.mean(), geometry=pointM, scale=scale).getInfo()
        slopeP = slope.reduceRegion(reducer=ee.Reducer.mean(), geometry=pointM, scale=scale).getInfo()

        df.at[siteId, 'elevMean'] = elevM['elevation']
        df.at[siteId, 'slopeMean'] = slopeM['slope']
        df.at[siteId, 'elevPoint'] = elevP['elevation']
        df.at[siteId, 'slopePoint'] = slopeP['slope']

        print(
            '{} {} {} {:.2f} {:.2f}'.format(
                k, siteId, datetime.now().strftime("%H:%M:%S"), time.time() - t0, time.time() - t1
            )
        )
        if k % 20 == 0:
            df.to_csv(outFile)
    df.to_csv(outFile)
