
import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 2, 3])
fig, ax = plt.subplots(1, 1)
ax.plot(a)
fig.show()


import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

# Save the plot by calling plt.savefig() BEFORE plt.show()
plt.savefig('coastlines.pdf')
plt.savefig('coastlines.png')

plt.show()