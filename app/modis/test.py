from pyhdf.SD import SD, SDC
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime as dt


# temp data for testing
dataFolder = r'C:\Users\geofk\work\database\MODIS\h08v05'
fileLst = [f for f in os.listdir(dataFolder) if f[-3:] == 'hdf']
tLst = [dt.strptime(f.split('.')[1][1:], '%Y%j') for f in fileLst]

dt.strptime('%Y%j')


hdf = SD(fileName, SDC.READ)
# hdf.datasets()
# hdf.attributes()
fpar = hdf.select('Fpar_500m')[:, :].astype(np.float)
lai = hdf.select('Lai_500m')[:, :].astype(np.float)
fpar[fpar > 100] = np.nan
lai[lai > 100] = np.nan


plt.imshow(lai)
plt.show()
