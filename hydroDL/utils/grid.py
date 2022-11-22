import numpy as np


def crd2grid(y, x, fillMiss=True):
    """ convert a list of points into grid """
    ux, indX0, indX = np.unique(x, return_index=True, return_inverse=True)
    uy, indY0, indY = np.unique(y, return_index=True, return_inverse=True)
    minDx = np.min(ux[1:] - ux[0:-1])
    minDy = np.min(uy[1:] - uy[0:-1])
    maxDx = np.max(ux[1:] - ux[0:-1])
    maxDy = np.max(uy[1:] - uy[0:-1])
    if maxDx >= minDx * 2 and fillMiss is True:
        indMissX = np.where((ux[1:]-ux[0:-1]) > minDx*2)[0]
        insertX = (ux[indMissX+1]+ux[indMissX])/2
        ux = np.insert(ux, indMissX+1, insertX)
    if maxDy >= minDy * 2:
        raise Exception('TODO:skipped coloums')
    #     indMissY=np.where((uy[1:]-uy[0:-1])>minDy*2)
    #     raise Exception('skipped coloums or rows')
    uy = uy[::-1]
    ny = len(uy)
    indY = ny - 1 - indY
    return (uy, ux, indY, indX)


def array2grid(data, *, lat, lon, fillMiss=True):
    (uy, ux, indY, indX) = crd2grid(lat, lon, fillMiss=fillMiss)
    ny = len(uy)
    nx = len(ux)
    if data.ndim == 2:
        nt = data.shape[1]
        grid = np.full([ny, nx, nt], np.nan)
        grid[indY, indX, :] = data
    elif data.ndim == 1:
        grid = np.full([ny, nx], np.nan)
        grid[indY, indX] = data
    return grid, uy, ux


def adjustGrid(grid, *, lat, lon):
    '''
    adjust the grid direction to look right in imshow
    lat, lon - 1d vector 
    grid [#lat,#lon (#t)]
    '''
    lon[lon > 180] = lon[lon > 180] - 360
    iLon = np.argsort(lon)
    lon = lon[iLon]
    iLat = np.argsort(lat)[::-1]
    lat = lat[iLat]
    if grid is None:
        return lat, lon
    if len(grid.shape) == 2:
        grid = grid[:, iLon]
        grid = grid[iLat, :]
    elif len(grid.shape) == 3:
        grid = grid[:, iLon, :]
        grid = grid[iLat, :, :]
    return grid, lat, lon


def clipGrid(grid=None, lat=None, lon=None, t=None,
             latR=None, lonR=None, tR=None):
    '''
    latR, lonR: crd range, include ghost cell
    tR: date range include tR[0], not include tR[1]
    WARN half-open
    '''
    if lonR is None:
        indX1, indX2 = (0, None)
        outLon = lon
    else:
        indX1 = np.where(lon < lonR[0])[0][-1]
        indX2 = np.where(lon > lonR[1])[0][1]
        outLon = lon[indX1:indX2]
    if latR is None:
        indY1, indY2 = (0, None)
        outLat = lat
    else:
        indY1 = np.where(lat > latR[1])[0][-1]
        indY2 = np.where(lat < latR[0])[0][1]
        outLat = lat[indY1:indY2]
    if tR is None:
        indT1, indT2 = (0, None)
        outT = t
    else:
        indT1 = np.where(t >= tR[0])[0][0]
        indT2 = np.where(t <= tR[1])[0][-1]+1
        outT = t[indT1:indT2]
    if grid is None:
        outGrid = grid
    elif len(grid.shape) == 3:
        outGrid = grid[indY1:indY2, indX1:indX2, indT1:indT2]
    elif len(grid.shape) == 2:
        outGrid = grid[indY1:indY2, indX1:indX2]
    elif len(grid.shape) == 1:
        outGrid = grid[indT1:indT2]
    return outGrid, (outLat, outLon, outT)


def interpGrid():
    pass


def intersectGrid(lat1, lon1, lat2, lon2, ndigit=8):
    # for efficiency, 1 should be the smaller than 2
    # lat1=np.around(lat1,decimals=ndigit)
    indLst1 = list()
    indLst2 = list()
    for k in range(len(lat1)):
        ind = np.where((lat2 == lat1[k]) & (lon2 == lon1[k]))[0]
        if len(ind) == 1:
            indLst1.append(k)
            indLst2.append(ind[0])
        if len(ind) > 1:
            raise Exception('Repeated crd')
    return np.array(indLst1), np.array(indLst2)
