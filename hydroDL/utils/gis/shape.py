from shapely.geometry import Point, shape
import time
import numpy as np


def is_ascend(x): return np.all(x[:-1] < x[1:])


def is_descend(x): return np.all(x[:-1] > x[1:])


def pointInPoly(y, x, shapeLst):
    nGrid = len(y)
    nShape = len(shapeLst)
    indLst = [-99] * nGrid
    t0 = time.time()
    polyLst = [shape(shapeLst[k]) for k in range(nShape)]
    pointLst = [Point(x[k], y[k]) for k in range(nGrid)]
    for j in range(nGrid):
        p = pointLst[j]
        for i in range(nShape):
            polygon = polyLst[i]
            if p.within(polygon):
                indLst[j] = i
            # print('\t pixel {} shape {}'.format(j,i), end='\r')
        print('\t pixel {} {:.2f}%'.format(j, j / nGrid * 100), end='\r')
    print('total time {}'.format(time.time() - t0))
    return indLst


def gridMask(lat, lon, polygon, ns=4):
    if not is_ascend(lon) and is_descend(lat):
        raise ValueError('not ascended or descended')
    mask = np.zeros([len(lat), len(lon)])
    bb = polygon.bounds
    indX1 = np.where(lon < bb[0])[0][-1]
    indX2 = np.where(lon > bb[2])[0][0]
    indY1 = np.where(lat > bb[3])[0][-1]
    indY2 = np.where(lat < bb[1])[0][0]
    dx = (lon[indX2] - lon[indX1]) / (indX2 - indX1)
    dy = (lat[indY1] - lat[indY2]) / (indY2 - indY1)
    for i in range(indX1, indX2 + 1):
        for j in range(indY1, indY2 + 1):
            x1 = lon[i] - dx / 2
            x2 = lon[i] + dx / 2
            y1 = lat[j] + dy / 2
            y2 = lat[j] - dy / 2
            bLst = [
                Point(x1, y1).within(polygon),
                Point(x1, y2).within(polygon),
                Point(x2, y1).within(polygon),
                Point(x2, y2).within(polygon)
            ]
            if not any(bLst):
                mask[j, i] = 0
            elif all(bLst):
                mask[j, i] = 1
            else:
                xm = np.linspace(x1 + dx / 2 / ns, x2 - dx / 2 / ns, ns)
                ym = np.linspace(y1 - dy / 2 / ns, y2 + dy / 2 / ns, ns)
                z = 0
                for xx in xm:
                    for yy in ym:
                        if Point(xx, yy).within(polygon):
                            z = z + 1
                mask[j, i] = z / ns / ns
    return mask
