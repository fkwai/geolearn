
import hydroDL
from mpl_toolkits import basemap
import matplotlib.pyplot as plt
import numpy as np
import os

from pyhdf.SD import SD, SDC

folder = r'D:\data\GLASS\LAI\AVHRR\1981'
fileName = 'GLASS01B02.V40.A1981001.2019353.hdf'
hdf = SD(os.path.join(folder, fileName), SDC.READ)
field = 'LAI'
data = hdf.select(field)[:, :]

fig, ax = plt.subplots(1, 1)
mm = basemap.Basemap(llcrnrlat=-90, urcrnrlat=90,
                     llcrnrlon=-180, urcrnrlon=180,
                     projection='cyl', resolution='l', ax=ax)
mm.drawcoastlines()
lon = np.arange(-179.975, 180, 0.05)
lat = np.arange(89.975, -90, -0.05)
x, y = mm(lon, lat)
xx, yy = np.meshgrid(x, y)
cs = mm.pcolormesh(xx, yy, data, cmap='jet')
fig.show()
