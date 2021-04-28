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
dirG = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'output')
dirD = os.path.join(kPath.dirData, 'USGS', 'GLASS', 'Daily')
sdStr = '1982-01-01'
edStr = '2018-12-31'
tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))

for siteNo in siteNoLst[:1]:
    print(siteNo)
    dfV = GLASS.readBasin(siteNo, freq='R')
    dfVP = pd.DataFrame({'date': tR}).set_index('date').join(dfV)
    dfVP = dfVP.interpolate(method='cubicspline')
    dfVP.to_csv(os.path.join(dirD, siteNo))


fig, axes = plt.subplots(3, 1)
for k, var in enumerate(varLst):
    axplot.plotTS(axes[k], dfV.index, dfV[var], styLst='*', cLst='r')
    axplot.plotTS(axes[k], dfVP.index, dfVP[var], styLst='-', cLst='b')
fig.show()
