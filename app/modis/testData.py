from pyhdf.SD import SD, SDC
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime as dt

from hydroDL import utils


# temp data for testing
dataFolder = r'C:\Users\geofk\work\database\MODIS\MCD15A2H_h08v05'
fileAllLst = [f for f in sorted(os.listdir(dataFolder)) if f[-3:] == 'hdf']
tAllLst = [dt.strptime(f.split('.')[1][1:], '%Y%j').date() for f in fileAllLst]
tAllAry = np.array(tAllLst, dtype='datetime64')
sd = np.datetime64('2003-01-01')
ed = np.datetime64('2003-03-01')
indFile = np.where((tAllAry >= sd) & (tAllAry <= ed))[0]
tAry = tAllAry[indFile]
fileLst = [fileAllLst[k] for k in indFile]

nt = tAry.shape[0]
matFpar = np.full([2400, 2400, nt], np.nan, dtype=np.int)
matLai = np.full([2400, 2400, nt], np.nan, dtype=np.int)
for k, fileName in enumerate(fileLst):
    filePath = os.path.join(dataFolder, fileName)
    hdf = SD(filePath, SDC.READ)
    fpar = hdf.select('Fpar_500m')[:, :]
    lai = hdf.select('Lai_500m')[:, :]
    matFpar[:, :, k] = fpar
    matLai[:, :, k] = lai

# fpar = fpar.astype(np.float)
# fpar[fpar > 100] = np.nan
# fpar = fpar*0.01
# lai = lai.astype(np.float)
# lai[lai > 100] = np.nan
# lai = lai*0.1
