import hydroDL
from hydroDL.data import dbCsv
from hydroDL.utils import gis, grid
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL import kPath
import time
import csv
import os
import pandas as pd
import numpy as np
from hydroDL.data import dbBasin


dirEco = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
fileEco = os.path.join(dirEco, 'basinRegion')
dfEco = pd.read_csv(fileEco, dtype={'siteNo': str}).set_index('siteNo')
fileEcoB = os.path.join(dirEco, 'basinRegionB')
dfEcoB = pd.read_csv(fileEcoB, dtype={'siteNo': str}).set_index('siteNo')

# Q90 sites
dataName = 'Q90'
dm = dbBasin.DataModelFull(dataName)
siteNoLst = dm.siteNoLst
df1 = dfEco.loc[siteNoLst]
df2 = dfEcoB.loc[siteNoLst]


dfT = pd.DataFrame(index=siteNoLst, columns=['Eco', 'EcoB'])
dfT['Eco']=df1['region']
dfT['EcoB']=df2['region']

dfT[dfT['Eco'] != dfT['EcoB']]
