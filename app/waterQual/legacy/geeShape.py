import ee
ee.Initialize()
import os
import time
import numpy as np
import pandas as pd
import datetime as dt
import fiona
from hydroDL.data import usgs, gee
from hydroDL.utils import time as utilTime
""" 
extract forcing data from google earth using shapefiles
"""

workDir = r'C:\Users\geofk\work\waterQuality'
modelDir = os.path.join(workDir, 'modelUsgs2')
fileSel = os.path.join(modelDir, 'siteNoSel')
outShapeFile = os.path.join(modelDir, 'basinSel_prj.shp')

shapeAll = fiona.open(outShapeFile)
datasetName = 'OREGONSTATE/PRISM/AN81d'
fieldLst = ['ppt', 'tmean', 'tdmean']
sd = pd.to_datetime('1981-01-01')
ed = pd.to_datetime('2019-12-20')
imageCol = ee.ImageCollection(datasetName).filterDate(
    gee.geeutils.t2ee(sd), gee.geeutils.t2ee(ed)).filterBounds(geog).select(
        fieldLst).sort('system:time_start')

errLst = list()
taskLst = list()
featLst = list()

errLst = list()
for k in range(0, 5):
    t0 = time.time()
    shape = shapeAll[k]
    siteNo = shape['properties']['GAGE_ID']
    crd = shape['geometry']['coordinates']
    geog = ee.Geometry.Polygon(crd)
    imageStack = imageCol.toBands()
    temp = imageStack.reduceRegion(ee.Reducer.mean(), geog, bestEffort=True)
    # getInfo
    try:
        out = temp.getInfo()
        v = out.values()
        data = np.reshape(np.array(list(out.values())), [-1, 3])
        saveFile = os.path.join(workDir, 'USGS', 'PRISM', siteNo)
        pdfOut = pd.DataFrame(data=data, columns=fieldLst)
        pdfOut['datetime'] = pd.to_datetime([x[:8] for x in out.keys()][::3])
        pdfOut.to_csv(saveFile, index=False)
        print('{} site {} time {}'.format(k, siteNo, time.time() - t0))
    except:
        errLst.append(k)
print(errLst)