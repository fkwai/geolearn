from hydroDL.data import dbBasin
import pandas as pd
from hydroDL import kPath, utils
import numpy as np
import os
dataName = 'QN90ref'
DF = dbBasin.DataFrameBasin(dataName)

dirOld = r'C:\Users\geofk\work\waterQuality\trainDataFull\{}'.format('Q90ref')

Q = np.load(os.path.join(dirOld, 'Q.npy'))
F = np.load(os.path.join(dirOld, 'F.npy'))

np.nansum(abs(DF.q - Q))
np.nansum(abs(DF.f - F))