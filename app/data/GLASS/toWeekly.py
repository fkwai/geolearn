import os
import numpy as np
import pandas as pd
from hydroDL import kPath
import shapefile
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL.data import GLASS

# read shapefiles to get siteNoLst
shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basinAll_prj.shp')
sf = shapefile.Reader(shpFile)
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]

# interploate by cubic-spline
varLst = ['LAI', 'FAPAR', 'NPP']
dirD = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'Daily')
dirW = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'Weekly')
sdStr = '1982-01-01'
edStr = '2018-12-31'
tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))

for siteNo in siteNoLst:
    print(siteNo)
    dfD = GLASS.readBasin(siteNo, freq='D')
    dfW = dfD.resample('W-TUE').mean()
    dfW.to_csv(os.path.join(dirW, siteNo))
