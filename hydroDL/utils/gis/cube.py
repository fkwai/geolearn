import iris
from iris.coords import DimCoord
from iris.cube import Cube
import numpy as np
import time


def getIntp(latS, lonS, latT, lonT):
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
    return intp


def intpCube(data, lat, lon, intp):
    t0 = time.time()
    latCube = DimCoord(lat, standard_name='latitude', units='degrees')
    lonCube = DimCoord(lon, standard_name='longitude', units='degrees')
    latCube.guess_bounds()
    lonCube.guess_bounds()
    cube = Cube(data, dim_coords_and_dims=[(latCube, 0), (lonCube, 1)])
    outCube = intp(cube)
    out = outCube.data
    print('interpolation time {:.2f}'.format(time.time()-t0))
    return out


def intp(dataS, latS, lonS, dataT, latT, lonT):
    t0 = time.time()
    lat1 = DimCoord(latS, standard_name='latitude', units='degrees')
    lon1 = DimCoord(lonS, standard_name='longitude', units='degrees')
    lat2 = DimCoord(latT, standard_name='latitude', units='degrees')
    lon2 = DimCoord(lonT, standard_name='longitude', units='degrees')
    lat1.guess_bounds()
    lon1.guess_bounds()
    lat2.guess_bounds()
    lon2.guess_bounds()
    cube1 = Cube(dataS, dim_coords_and_dims=[(lat1, 0), (lon1, 1)])
    cube2 = Cube(dataT, dim_coords_and_dims=[(lat2, 0), (lon2, 1)])
    cubeOut = cube1.regrid(cube2, iris.analysis.AreaWeighted(mdtol=0))
    print('interpolation time {:.2f}'.format(time.time()-t0))
    return cubeOut.data
