from iris.analysis import geometry
import iris
from iris.cube import Cube
from iris.coords import DimCoord
import numpy as np
import shapefile
from shapely.geometry import shape
import os
import time
import argparse
import matplotlib.pyplot as plt

import hydroDL.data.gridMET.io as io
from hydroDL.utils import gis
from hydroDL import kPath
import time
# mask generating
# test efficiency, validate result

ncFile = os.path.join(kPath.dirRaw, 'gridMET', 'etr_1979.nc')
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basin_CONUS_prj.shp')
saveDir = os.path.join(kPath.dirData, 'USGS', 'gridMET', 'mask')

t, lat, lon = io.readNcInfo(ncFile)
sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]
area = np.array([rec[0] for rec in recLst])

np.where(area>4e10)[0]

geog = shape(shapeLst[7125])
t0 = time.time()
mask = gis.gridMask(lat, lon, geog)
print('old {:.2f}'.format(time.time()-t0))

# cube method
t0 = time.time()
latCube = DimCoord(lat, standard_name='latitude', units='degrees')
lonCube = DimCoord(lon, standard_name='longitude', units='degrees')
cube = Cube(np.zeros([len(lat), len(lon)]),
            dim_coords_and_dims=[(latCube, 0), (lonCube, 1)])
latCube.guess_bounds()
lonCube.guess_bounds()
mask2 = geometry.geometry_area_weights(cube, geog)
print('cube {:.2f}'.format(time.time()-t0))


[x, y] = geog.exterior.coords.xy
lonM, latM = np.meshgrid(lon, lat)

fig, axes = plt.subplots(2, 1)
axes[0].plot(x, y, 'k-')
axes[0].plot(lonM[mask > 0], latM[mask > 0], 'bo')
axes[0].plot(lonM[mask == 1], latM[mask == 1], 'ro')
axes[1].plot(x, y, 'k-')
axes[1].plot(lonM[mask > 0], latM[mask > 0], 'bo')
axes[1].plot(lonM[mask == 1], latM[mask == 1], 'ro')
fig.show()

