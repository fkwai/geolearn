import hydroDL
from hydroDL.data import dbCsv
from hydroDL.utils import grid
import importlib

import numpy as np
# read database crd
importlib.reload(dbCsv)
rootDB = hydroDL.pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUS', tRange=tRange)
lat, lon = df.getGeo()

yt = df.getDataTs('SMAP_AM')
ecoL1 = df.getDataConst('ecoRegionL1')

ind = np.where(ecoL1 == 5)[0]

# write a subset file
varC = ['ecoRegionL'+str(x) for x in [1, 2, 3]]
subset = 'CONUSv2f1'
df.subsetData(subset, varC=varC)
