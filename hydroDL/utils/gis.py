from shapely.geometry import Point, shape
import time


def pointInPoly(lat, lon, shapeLst):
    nGrid = len(lat)
    nShape = len(shapeLst)
    indLst = [-99]*nGrid
    t0 = time.time()
    polyLst = [shape(shapeLst[x]) for x in range(nShape)]
    pointLst = [Point(lon[x], lat[x]) for x in range(nGrid)]
    for j in range(nGrid):
        p = pointLst[j]
        for i in range(nShape):
            polygon = polyLst[i]
            if p.within(polygon):
                indLst[j] = i
            # print('\t pixel {} shape {}'.format(j,i), end='\r')
        print('\t pixel {} {:.2f}%'.format(j, j/nGrid*100), end='\r')
    print('total time {}'.format(time.time()-t0))
    return indLst
