from mpl_toolkits import basemap
import numpy as np
import matplotlib.pyplot as plt


def base(ax, bounding, prj='cyl'):
    mm = basemap.Basemap(llcrnrlat=bounding[0],
                         urcrnrlat=bounding[1],
                         llcrnrlon=bounding[2],
                         urcrnrlon=bounding[3],
                         projection=prj,
                         resolution='c',
                         ax=ax)
    mm.drawcoastlines()
    # mm.drawstates(linestyle='dashed')
    # mm.drawcountries(linewidth=1.0, linestyle='-.')
    return mm


def grid(mm, lat, lon, data, *, cmap=plt.cm.jet, vmin=None, vmax=None):
    bounding = [np.min(lat), np.max(lat), np.min(lon), np.max(lon)]
    x, y = mm(lon, lat)
    xx, yy = np.meshgrid(x, y)
    mm.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)


def plotMap(data, *, ax=None, lat=None, lon=None, title=None, cRange=None,
            shape=None, pts=None, figsize=(8, 4), cbar=True, cmap=plt.cm.jet,
            bounding=None, prj='cyl', returnAll=False, plotPoint=False):

    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        if data is not None:
            temp = flatData(data)
            vmin = np.percentile(temp, 5)
            vmax = np.percentile(temp, 95)
        else:
            (vmin, vmax) = (None, None)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    if len(data.squeeze().shape) == 1 or plotPoint is True:
        isGrid = False
    else:
        isGrid = True
    if bounding is None:
        bounding = [np.min(lat), np.max(lat), np.min(lon), np.max(lon)]

    mm = basemap.Basemap(llcrnrlat=bounding[0],
                         urcrnrlat=bounding[1],
                         llcrnrlon=bounding[2],
                         urcrnrlon=bounding[3],
                         projection=prj,
                         resolution='c',
                         ax=ax)
    mm.drawcoastlines()
    mm.drawstates(linestyle='dashed')
    #     mm.drawcountries(linewidth=1.0, linestyle='-.')

    x, y = mm(lon, lat)
    if isGrid is True:
        xx, yy = np.meshgrid(x, y)
        cs = mm.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)
        # cs = mm.imshow(
        #     np.flipud(data),
        #     cmap=plt.cm.jet(np.arange(0, 1, 0.1)),
        #     vmin=vmin,
        #     vmax=vmax,
        #     extent=[x[0], x[-1], y[0], y[-1]])
    else:
        cs = mm.scatter(x,
                        y,
                        c=data,
                        cmap=cmap,
                        s=10,
                        marker='*',
                        vmin=vmin,
                        vmax=vmax)

    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                mm.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            mm.plot(x, y, color='r', linewidth=3)
    if pts is not None:
        mm.plot(pts[1], pts[0], 'k*', markersize=4)
        npt = len(pts[0])
        for k in range(npt):
            plt.text(pts[1][k],
                     pts[0][k],
                     string.ascii_uppercase[k],
                     fontsize=18)
    if cbar is True:
        mm.colorbar(cs, location='bottom', pad='5%')

    if title is not None:
        ax.set_title(title)
    if returnAll is True:
        return fig, ax, mm, cs
    else:
        return ax
