import ee
import os
import time
import numpy as np
import pandas as pd
import datetime as dt
from hydroDL.data import usgs, gee

workDir = r'C:\Users\geofk\work\waterQuality'
siteNo = '04086120'

# find out site
dfSite = usgs.readUsgsText(os.path.join(workDir, 'inventory_NWIS_streamflow'))
rowSite = dfSite.loc[dfSite['site_no'] == siteNo]
lat = rowSite['dec_lat_va'].values[0]
lon = rowSite['dec_long_va'].values[0]
sd = dt.datetime.strptime(rowSite['qw_begin_date'].values[0], '%Y-%m-%d')
ed = dt.datetime.strptime(rowSite['qw_end_date'].values[0], '%Y-%m-%d')
# ed = sd + dt.timedelta(days=25)
sd=dt.datetime(year=2015,month=9,day=23)
ed=dt.datetime(year=2015,month=9,day=30)

point = ee.Geometry.Point(lon, lat)
region = point.buffer(100)
fieldLst = ['ppt', 'tmean', 'tdmean']

nd = 50
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
savePath = os.path.join(os.path.join(workDir, 'data', 'forcing', siteNo+'temp'))
out.to_csv(savePath, index=False)
