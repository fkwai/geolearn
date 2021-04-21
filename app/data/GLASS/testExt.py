from hydroDL.utils import gis
from shapely.geometry import shape
import shapefile
import time
from hydroDL import kPath
from pyhdf.SD import SD, SDC
import os
import matplotlib.pyplot as plt
import numpy as np

# both mask and data are global. Extract using a subset

folder = r'D:\data\GLASS\LAI\AVHRR\1981'
fileName = 'GLASS01B02.V40.A1981001.2019353.hdf'
hdf = SD(os.path.join(folder, fileName), SDC.READ)
field = 'LAI'
data = hdf.select(field)[:, :]

lon = np.arange(-179.975, 180, 0.05)
lat = np.arange(89.975, -90, -0.05)

np.where(np.isclose(lat, 24.975))
np.where(np.isclose(lat, 49.975))
np.where(np.isclose(lon, -124.975))
np.where(np.isclose(lon, -64.975))


fig, ax = plt.subplots(1, 1)
ax.imshow(data[800:1300,1100:2300])
fig.show()
