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


# load location
crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd = pd.read_csv(crdFile, index_col=0)
ee.Initialize()

sd = '2015-01-01'
ed = '2023-03-01'

colLFMC = ee.ImageCollection("users/kkraoj/lfm-mapper/lfmc_col_25_may_2021").filterDate(sd, ed)
proj = colLFMC.first().projection().getInfo()
scale = colLFMC.first().projection().nominalScale().getInfo()


def getMean(image, region, scale):
    date = image.date().format('YYYY-MM-dd')
    mean_values = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=scale, maxPixels=1e9)
    return ee.Feature(None, mean_values.set('date', date))


def feature2df(data):
    dataLst = []
    for feature in data['features']:
        properties = feature['properties']
        dataLst.append(properties)
    df = pd.DataFrame(dataLst)
    df = df.set_index('date')
    return df


outFolder = os.path.join(kPath.dirVeg, 'Raoj')
t0 = time.time()

for k, siteId in enumerate(tabCrd.index.tolist()):
    outFile = os.path.join(outFolder, siteId + '.csv')
    if os.path.exists(outFile):
        continue
    lat = tabCrd.loc[siteId]['lat']
    lon = tabCrd.loc[siteId]['lon']
    if np.isnan(lat) or np.isnan(lon):
        continue
    point = ee.Geometry.Point([lon, lat])
    pointPrj = point.transform(proj['crs'])
    t1 = time.time()
    data = colLFMC.filterBounds(pointPrj).map(lambda image: getMean(image, pointPrj, scale)).getInfo()
    df = feature2df(data)
    df.to_csv(os.path.join(outFolder, siteId + '.csv'))
    print(
        '{} {} {} {:.2f} {:.2f}'.format(
            k, siteId, datetime.now().strftime("%H:%M:%S"), time.time() - t0, time.time() - t1
        )
    )
