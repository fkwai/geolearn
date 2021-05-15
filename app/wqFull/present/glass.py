from hydroDL import kPath
import os
from shapely.geometry import shape
import numpy as np
import shapefile
from mpl_toolkits import basemap
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC

# for a site
siteNo = '01184000'
# extract from arcgis
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'single', siteNo)

maskDir = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'mask')
maskFile = os.path.join(maskDir, siteNo)
mask = np.load(maskFile+'.npz')['mask']
lon = np.arange(-179.975, 180, 0.05)
lat = np.arange(89.975, -90, -0.05)

# mask
fig, ax = plt.subplots(1, 1)
mm = basemap.Basemap(llcrnrlat=25, urcrnrlat=50,
                     llcrnrlon=-125, urcrnrlon=-65,
                     projection='cyl', resolution='c', ax=ax)

mm.drawcoastlines()
x, y = mm(lon, lat)
xx, yy = np.meshgrid(x, y)
cs = mm.pcolormesh(xx, yy, mask, cmap='viridis', vmin=0, vmax=1)
mm.readshapefile(shpFile, siteNo, linewidth=1,color='r')
mm.colorbar(cs, location='bottom', pad='5%')
fig.show()

# data
folder = r'D:\data\GLASS\LAI\AVHRR\1981'
fileName = 'GLASS01B02.V40.A1981001.2019353.hdf'
hdf = SD(os.path.join(folder, fileName), SDC.READ)
field = 'LAI'
data = hdf.select(field)[:, :]

fig, ax = plt.subplots(1, 1)
mm = basemap.Basemap(llcrnrlat=25, urcrnrlat=50,
                     llcrnrlon=-125, urcrnrlon=-65,
                     projection='cyl', resolution='c', ax=ax)
mm.drawcoastlines()
x, y = mm(lon, lat)
xx, yy = np.meshgrid(x, y)
cs = mm.pcolormesh(xx, yy, data, cmap='viridis', vmin=0, vmax=100)
mm.colorbar(cs, location='bottom', pad='5%')
fig.show()
