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

productM = 'MODIS/061/MCD43A4'
colM = ee.ImageCollection(productM).filterDate(sd, ed)
opt='modisgrid'

if opt == 'nadgrid':
    proj = ee.Projection('EPSG:5072').getInfo()
elif opt == 'modisgrid':
    proj = colM.first().projection().getInfo()
    dxM = proj['transform'][0]
    dyM = proj['transform'][4]
    

latAll=tabCrd['lat'].values
lonAll=tabCrd['lon'].values

lat=latAll[0]
lon=lonAll[0]   
point = ee.Geometry.Point([lon, lat])
pointM = point.transform(proj['crs'])
x, y = pointM.getInfo()['coordinates']
xmod = np.mod(x, dxM)
ymod = np.mod(y, dyM)
x1 = x - xmod
y2 = y - ymod
x2 = x1 + dxM
y1 = y2 + dyM
bb = ee.Geometry.Rectangle([x1, y1, x2, y2], proj['crs'], False)

nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD')
nlcd2016 = nlcd.filter(ee.Filter.eq('system:index', '2016')).first().select('landcover')
scaleNLCD = nlcd2016.projection().nominalScale().getInfo()
data = nlcd2016.reduceRegion(
    reducer=ee.Reducer.toList(), geometry=bb, scale=scaleNLCD).getInfo()

