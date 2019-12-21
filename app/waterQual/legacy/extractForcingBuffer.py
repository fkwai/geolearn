import ee
ee.Initialize()
import os
import time
import numpy as np
import pandas as pd
import datetime as dt
from hydroDL.data import usgs, gee

workDir = r'C:\Users\geofk\work\waterQuality'
# siteNo = '04086120'
siteNo = '053416972'

# find out site
dfSite = usgs.readUsgsText(os.path.join(workDir, 'inventory_NWIS_streamflow'))
rowSite = dfSite.loc[dfSite['site_no'] == siteNo]
lat = rowSite['dec_lat_va'].values[0]
lon = rowSite['dec_long_va'].values[0]
# not right somehow
# sd = dt.datetime.strptime(rowSite['qw_begin_date'].values[0], '%Y-%m-%d')
# ed = dt.datetime.strptime(rowSite['qw_end_date'].values[0], '%Y-%m-%d')
dfDaily = usgs.readUsgsText(os.path.join(workDir, 'data', 'dailyTS', siteNo),
                            dataType='dailyTS')
sd = pd.to_datetime(dfDaily['datetime'].values[0])
ed = pd.to_datetime(dfDaily['datetime'].values[-1])

# (lon, lat) = (-92.4363056, 45.12002778)
# (lon, lat) = (-88.4008333, 43.62444444)
# (lon, lat) = (-88.4063056, 43.12002778)

point = ee.Geometry.Point(lon, lat)
region = point.buffer(100)
datasetName = 'OREGONSTATE/PRISM/AN81d'
fieldLst = ['ppt', 'tmean', 'tdmean']
# datasetName = 'NASA/NLDAS/FORA0125_H002'
# fieldLst=['total_precipitation','temperature']

# test of gee server
imageCol = ee.ImageCollection(datasetName).filterDate(
    gee.utils.t2ee(sd), gee.utils.t2ee(ed)).filterBounds(point).select(
        fieldLst).sort('system:time_start')
imageLst = imageCol.toList(5)
image = ee.Image(imageLst.get(0))
temp = image.reduceRegion(ee.Reducer.median(), region).getInfo()
print(temp)
 
nd = 10
t1 = sd
if (ed - sd).days > nd:
    t2 = t1 + dt.timedelta(days=nd)
else:
    t2 = ed

dfLst = list()
while t1 < t2 <= ed:
    t0 = time.time()
    imageCol = ee.ImageCollection("OREGONSTATE/PRISM/AN81d").filterDate(
        gee.utils.t2ee(t1), gee.utils.t2ee(t2)).filterBounds(point).select(
            fieldLst).sort('system:time_start')
    df = gee.getRegion(imageCol, fieldLst, region)
    dfLst.append(df)
    t1 = t2
    t2 = t1 + dt.timedelta(days=nd)
    if t2 > ed:
        t2 = ed
    print('downloaded {}/{} time {}'.format((t1 - sd).days, (ed - sd).days,
                                            time.time() - t0))
out = pd.concat(dfLst)
out = out.sort_values(by=['datetime'])
savePath = os.path.join(
    os.path.join(workDir, 'data', 'forcing', siteNo + 'temp'))
out.to_csv(savePath, index=False)
