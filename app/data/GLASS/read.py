from pyhdf.SD import SD, SDC
import os
import matplotlib.pyplot as plt
import numpy as np

folder = r'D:\data\GLASS\LAI\AVHRR\1981'
fileName = 'GLASS01B02.V40.A1981001.2019353.hdf'

hdf = SD(os.path.join(folder, fileName), SDC.READ)
field = 'LAI'
data = hdf.select(field)[:, :]

fig, ax = plt.subplots(1, 1)
ax.imshow(data)
fig.show()

lon = np.arange(-179.95, 180, 0.1)
lat = np.arange(89.95, -90, -0.1)
