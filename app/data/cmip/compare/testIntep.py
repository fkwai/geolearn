
import iris
import hydroDL.utils.ts
from scipy import interpolate
import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.data.cmip.io
import hydroDL.data.gridMET.io
import importlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
from iris.coords import DimCoord
from iris.cube import Cube

importlib.reload(hydroDL.data)

# set dates
d1 = np.datetime64('2013-01-01')
d2 = np.datetime64('2015-01-01')
d3 = np.datetime64('2017-01-01')
y1, y2, y3 = [d.astype(object).year for d in [d1, d2, d3]]
latR = [25, 50]
lonR = [-125, -65]
varG = 'pr'
varC = 'pr'
func = 'nansum'

# read gridMet
gridF1, (latG, lonG, tG1) = hydroDL.data.gridMET.io.read(
    varG, y1, y2, dtype='float32')

# read CMIP6
df = hydroDL.data.cmip.io.walkFile()
modelName = 'MPI-ESM1-2-XR'
data1, latC1, lonC1, tC1 = hydroDL.data.cmip.io.readCMIP(
    dfFile=df, var=varC, exp='hist-1950', latR=latR, lonR=lonR,
    sd=d1, ed=d2, model=modelName)


latT = latC1
lonT = lonC1
latS = latG
lonS = lonG

lat1 = DimCoord(latS, standard_name='latitude', units='degrees')
lon1 = DimCoord(lonS, standard_name='longitude', units='degrees')
lat2 = DimCoord(latT, standard_name='latitude', units='degrees')
lon2 = DimCoord(lonT, standard_name='longitude', units='degrees')

lat1.guess_bounds()
lon1.guess_bounds()
lat2.guess_bounds()
lon2.guess_bounds()

cube1 = Cube(np.zeros([len(latS), len(lonS)]),
             dim_coords_and_dims=[(lat1, 0), (lon1, 1)])
cube2 = Cube(np.zeros([len(latT), len(lonT)]),
             dim_coords_and_dims=[(lat2, 0), (lon2, 1)])
intp = iris.analysis.AreaWeighted(mdtol=0).regridder(cube1, cube2)

lat1.is_contiguous()
lat2.is_contiguous()
lon1.is_contiguous()
lon2.is_contiguous()

lat = latG
lon = lonG
data = gridF1
latCube = DimCoord(lat, standard_name='latitude', units='degrees')
lonCube = DimCoord(lon, standard_name='longitude', units='degrees')
latCube.guess_bounds()
lonCube.guess_bounds()
cube = Cube(data, dim_coords_and_dims=[(latCube, 0), (lonCube, 1)])
outCube = intp(cube)
out=outCube.data
fig, ax = plt.subplots(1, 1)
ax.imshow(out[:, :, 0])
fig.show()
