

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from hydroDL import utils

'''
# have to input through gridspec
import matplotlib.gridspec as gridspec
fig = plt.figure()
gsM = gridspec.GridSpec(1, 1)
mapplot.mapPoint(fig, gs, lat, lon, data)

'''


def mapPoint(fig, gs, lat, lon, data,
             vRange=None, cmap='jet', s=30, marker='o',
             cb=True, centerZero=False):
    if np.isnan(data).all():
        print('all nan in data')
        return
    if vRange is None:
        vmin, vmax = utils.vRange(data, centerZero=centerZero)
    else:
        vmin, vmax = vRange

    ax = fig.add_subplot(gs, projection=ccrs.PlateCarree())
    ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='auto', color='k')
    ind = np.where(~np.isnan(data))[0]
    cs = ax.scatter(lon[ind], lat[ind], c=data[ind], cmap=cmap,
                    s=s, marker=marker, vmin=vmin, vmax=vmax)
    if cb is True:
        plt.colorbar(cs, orientation="horizontal")
    return ax


'''
def mapPointClass(ax, lat, lon, data,
                  vLst=None, cLst=None, mLst=None, labLst=None,
                  labelCount=True):
    dataP, latP, lonP = utils.rmNan([data, lat, lon], returnInd=False)
    if vLst is None:
        vLst = list(np.unique(dataP))
    if cLst is None:
        cLst = plt.cm.jet(np.linspace(0, 1, len(vLst)))
    if labLst is None:
        labLst = vLst
    if mLst is None:
        mLst = ['*' for v in vLst]
    mm = basemap.Basemap(llcrnrlat=25, urcrnrlat=50,
                         llcrnrlon=-125, urcrnrlon=-65,
                         projection='cyl', resolution='c', ax=ax)
    mm.drawcoastlines()
    mm.drawcountries(linestyle='dashed')
    mm.drawstates(linestyle='dashed', linewidth=0.5)
    for k, v in enumerate(vLst):
        ind = np.where(dataP == v)[0]
        label = labLst[k]
        if labelCount is True:
            label = label + ' '+str(len(ind))
        mm.plot(lonP[ind], latP[ind], c=cLst[k], label=label,
                marker=mLst[k], ls='None')
    ax.legend()
    return mm


def mapGrid(ax, lat, lon, data, vRange=None, cmap='jet'):
    if np.isnan(data).all():
        print('all nan in data')
        return
    if vRange is None:
        vmin = np.percentile(data[~np.isnan(data)], 10)
        vmax = np.percentile(data[~np.isnan(data)], 90)
    else:
        vmin, vmax = vRange
    mm = basemap.Basemap(llcrnrlat=25, urcrnrlat=50,
                         llcrnrlon=-125, urcrnrlon=-65,
                         projection='cyl', resolution='c', ax=ax)
    mm.drawcoastlines()
    mm.drawcountries(linestyle='dashed')
    mm.drawstates(linestyle='dashed', linewidth=0.5)
    x, y = mm(lon, lat)
    xx, yy = np.meshgrid(x, y)
    cs = mm.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)
    mm.colorbar(cs, location='bottom', pad='5%')
    return mm
'''
