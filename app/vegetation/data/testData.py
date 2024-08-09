import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import matplotlib.pyplot as plt

rho = 45
dataName = 'singleDaily'
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)

data = df.xc[:, 6:15]
var = df.varXC[6:15]
print(var)
dataMax = np.nanmax(data, axis=1)


sorted_data = np.sort(dataMax)[::-1]
cdf = np.arange(1, len(sorted_data) + 1) 
fig, ax = plt.subplots(1, 1)
ax.plot(sorted_data,cdf)
fig.show()

np.sum([dataMax>0.8])/len(dataMax)