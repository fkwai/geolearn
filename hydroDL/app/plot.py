from mpl_toolkits import basemap
import matplotlib.pyplot as plt
import numpy as np


def plotMap(ax, lat, lon, data, dataN, title=None, vRange=None):
    if np.isnan(data).all():
        print('all nan in data')
        return
    if vRange is None:
        vmin = np.percentile(data[~np.isnan(data)], 10)
        vmax = np.percentile(data[~np.isnan(data)], 90)
    else:
        vmin, vmax = vRange
    mm = Basemap(llcrnrlat=25, urcrnrlat=50,
                 llcrnrlon=-125, urcrnrlon=-65,
                 projection='cyl', resolution='c', ax=ax)
    mm.drawcoastlines()
    mm.drawstates(linestyle='dashed')
    ind1 = np.where(dataN >= 10)
    ind2 = np.where(dataN < 10)
    cs = mm.scatter(lon[ind1], lat[ind1], c=data[ind1], cmap=plt.cm.jet,
                    s=80, marker='.', vmin=vmin, vmax=vmax)
    cs = mm.scatter(lon[ind2], lat[ind2], c=data[ind2], cmap=plt.cm.jet,
                    s=30, marker='*', vmin=vmin, vmax=vmax)
    mm.colorbar(cs, location='bottom', pad='5%')
    ax.set_title(title)
