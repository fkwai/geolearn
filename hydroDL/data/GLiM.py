# import fiona
import pandas as pd
import numpy as np
import os
import json
from hydroDL import kPath
import geopandas as gpd
import time

dataCode = [
    ['nd', 'No Data', 0],
    ['su', 'Unconsolidated sediments', 1],
    ['ss', 'Siliciclastic sedimentary rocks', 2],
    ['py', 'Pyroclastics', 3],
    ['sm', 'Mixed sedimentary rocks', 4],
    ['sc', 'Carbonate sedimentary rocks', 5],  # Ca Mg
    ['ev', 'Evaporites', 6],  # Ca
    ['va', 'Acid volcanic rocks', 7],
    ['vi', 'Intermediate volcanic rocks', 8],
    ['vb', 'Basic volcanic rocks', 9],
    ['pa', 'Acid plutonic rocks', 10],
    ['pi', 'Intermediate plutonic rocks', 11],
    ['pb', 'Basic plutonic rocks', 12],
    ['mt', 'Metamorphics', 13],
    ['wb', 'Water Bodies', 14],
    ['ig', 'Ice and Glaciers', 15]]

combineLst = [
    [1, 2],
    [4, 5],
    [1, 2, 4],
    [7, 8, 10],
    [7, 8, 10, 11],
    [9, 12]
]


tabCode = pd.DataFrame(columns=['xx', 'desc', 'code'], data=dataCode)
tabCode = tabCode.set_index('xx')


def updateCode():
    # udpate code of the GLiM in .shp
    # projection -> clip -> to raster in ArcGIS
    dirGlim = os.path.join(kPath.dirData, 'GLiM')
    shapeFile = os.path.join(dirGlim, 'GLiM_NA.shp')
    shapeFileOut = os.path.join(dirGlim, 'GLiM_NA_code.shp')

    shape = fiona.open(shapeFile)
    meta = shape.meta
    meta['schema']['properties']['code'] = 'int'
    t0 = time.time()
    n = len(shape)
    with fiona.open(shapeFileOut, 'w', **meta) as output:
        for k, feat in enumerate(shape):
            if k % 1000 == 0:
                print('{:.2f}% {:.2f}'.format(k/n*100, time.time()-t0))
            xx = feat['properties']['xx']
            code = tabLookup.loc[xx]['code']
            feat['properties']['code'] = int(code)
            output.write(feat)

    # quick test using geopandas
    # gdf = gpd.read_file(shapeFileOut)
