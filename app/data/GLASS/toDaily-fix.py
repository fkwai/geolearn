import os
import numpy as np
import pandas as pd
from hydroDL import kPath
import shapefile
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL.data import GLASS
import json

dirG = os.path.join(kPath.dirUsgs, 'GLASS', 'output')
dirD = os.path.join(kPath.dirUsgs, 'GLASS', 'Daily')

# Select sites
# fileSiteNo = os.path.join(kPath.dirUsgs, 'basins', 'siteNoLst.json')
# with open(fileSiteNo) as fp:
#     dictSite = json.load(fp)
# siteNoLstAll = dictSite['CONUS']
# siteNoLstTemp = [f for f in sorted(os.listdir(dirD))]
# siteNoLst = [f for f in siteNoLstAll if f not in siteNoLstTemp]

# all sites in dirG
siteNoLst = [f for f in sorted(os.listdir(dirG))]

# interploate by cubic-spline
varLst = ['LAI', 'FAPAR', 'NPP']
# varLst = ['LAI']
sdStr = '1982-01-01'
edStr = '2018-12-31'
tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))

for siteNo in siteNoLst:
    print(siteNo)
    dfV = GLASS.readBasin(siteNo, freq='R',varLst=varLst)
    dfVP = pd.DataFrame({'date': tR}).set_index('date').join(dfV)
    dfVP = dfVP.interpolate(method='cubicspline')
    dfVP.to_csv(os.path.join(dirD, siteNo))


fig, axes = plt.subplots(3, 1)
for k, var in enumerate(varLst):
    axplot.plotTS(axes[k], dfV.index, dfV[var], styLst='*', cLst='r')
    axplot.plotTS(axes[k], dfVP.index, dfVP[var], styLst='-', cLst='b')
fig.show()
