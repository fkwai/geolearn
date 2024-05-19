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
# crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
# tabCrd = pd.read_csv(crdFile, index_col=0)
outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_single.json')
with open(outFile, 'r') as fp:
    dictLst = json.load(fp)
dfSite = pd.DataFrame(columns=['siteId', 'siteName', 'state', 'fuel', 'gacc', 'lat', 'lon'])
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

productM = 'MODIS/061/MCD43A4'
productS = 'COPERNICUS/S1_GRD'
productL = 'LANDSAT/LC08/C02/T1_L2'
strS = 'sentinel1'
strM = 'MCD43A4'
strL = 'landsat8'


def qcModis(image):
    qcBandLst = ['BRDF_Albedo_Band_Mandatory_Quality_Band{}'.format(i) for i in range(1, 8)]
    qualityBands = image.select(qcBandLst)
    mask = qualityBands.reduce(ee.Reducer.max()).eq(0)
    return image.updateMask(mask)


def qcLandsat(image):
    mask1 = image.select('QA_PIXEL').bitwiseAnd(1 << 5).eq(0)
    mask2 = image.select('QA_PIXEL').bitwiseAnd(1 << 3).eq(0)
    mask = mask1.And(mask2)
    return image.updateMask(mask)


def getMean(image, region, scale):
    date = image.date().format('YYYY-MM-dd')
    mean_values = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=scale, maxPixels=1e9)
    return ee.Feature(None, mean_values.set('date', date))


def getMeanMulti(image, region, scale):
    mean_values = image.reduceRegions(collection=region, reducer=ee.Reducer.mean(), scale=scale)
    return mean_values.map(lambda f: f.set('date', image.date().format('YYYY-MM-dd')))


def feature2df(data):
    dataLst = []
    for feature in data['features']:
        properties = feature['properties']
        dataLst.append(properties)
    df = pd.DataFrame(dataLst)
    df = df.set_index('date')
    return df


colM = ee.ImageCollection(productM).filterDate(sd, ed).map(qcModis)
colL = ee.ImageCollection(productL).filterDate(sd, ed).map(qcLandsat)
colS = (
    ee.ImageCollection(productS)
    .filterDate(sd, ed)
    .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
)

optLst = ['nadgrid', 'modisgrid']

for opt in optLst:
    for strTemp in [strS, strM, strL]:
        outFolder = os.path.join(kPath.dirVeg, 'RS', '{}-{}'.format(strTemp, opt))
        if not os.path.exists(outFolder):
            os.mkdir(outFolder)
    if opt == 'nadgrid':
        proj = ee.Projection('EPSG:5072').getInfo()
    elif opt == 'modisgrid':
        projM = colM.first().projection().getInfo()
        dxM = projM['transform'][0]
        dyM = projM['transform'][4]

    t0 = time.time()
    for k, siteId in enumerate(dfSite.index.tolist()):
        # siteId=dfSite.index.tolist()[0]
        lat = dfSite.loc[siteId]['lat']
        lon = dfSite.loc[siteId]['lon']
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
            pointM = point.transform(projM['crs'])
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
        # download
        for col, strTemp, scale in zip([colM, colL, colS], [strM, strL, strS], [500, 30, 10]):
            outFolder = os.path.join(kPath.dirVeg, 'RS', '{}-{}'.format(strTemp, opt))
            fileName = os.path.join(outFolder, siteId + '.csv')
            if not os.path.exists(fileName):
                data = col.filterBounds(bb).map(lambda image: getMean(image, bb, scale)).getInfo()
                df = feature2df(data)
                df.to_csv(os.path.join(outFolder, siteId + '.csv'))
            print('{} {} {} {:.2f} {:.2f}'.format(k, siteId, strTemp, time.time() - t0, time.time() - t1))
