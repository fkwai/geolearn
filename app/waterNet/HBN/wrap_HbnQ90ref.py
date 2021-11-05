from hydroDL.data import dbBasin, gageII, gridMET, gageII
import pandas as pd
import os
from hydroDL import kPath
import matplotlib.pyplot as plt
import numpy as np

file1 = os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv')
dfHBN = pd.read_csv(file1, dtype={'siteNo': str}).set_index('siteNo')
s1 = dfHBN.index.tolist()
file2 = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel', 'Q90ref')
s2 = pd.read_csv(file2, header=None, dtype=str)[0].tolist()

siteNoLst = list(set(s1) & set(s2))
siteNoLst.remove('06879650')

# outShapeFile = os.path.join(kPath.dirData, 'USGS', 'basins', 'HBN36.shp')
# gageII.extractBasins(siteNoLst, outShapeFile)


dataName = 'HBN_Q90ref'
# DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, varG=gageII.varLstEx)
DF = dbBasin.DataFrameBasin(dataName)
DF.saveSubset('B10', ed='2009-12-31')
DF.saveSubset('A10', sd='2010-01-01')
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')

np.where(np.isnan(DF.q))

k = 0

fig, ax = plt.subplots(1, 1)
ax.plot(DF.t, DF.q[:, k, 1])
ax.set_title(DF.siteNoLst[k])
fig.show()
