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
fileEco = os.path.join(dirEco, 'basinEco')
dfEco = pd.read_csv(fileEco, dtype={'siteNo': str}).set_index('siteNo')
fileEcoB = os.path.join(dirEco, 'basinEcoB')
dfEcoB = pd.read_csv(fileEcoB, dtype={'siteNo': str}).set_index('siteNo')

# Q90 sites
dataName = 'Q90'
dm = dbBasin.DataModelFull(dataName)
siteNoLst = dm.siteNoLst
df1 = dfEco.loc[siteNoLst]
df2 = dfEcoB.loc[siteNoLst]

dfT = pd.DataFrame(index=siteNoLst, columns=['Eco', 'EcoB'])
dfT['Eco'] = df1[['code0', 'code1', 'code2']].agg('-'.join, axis=1)

varLst = ['code0', 'code1', 'code2']
for var in varLst:
    df1[var] = df1[var].astype(int).astype(str).str.zfill(2)
dfT['Eco'] = df1[varLst].agg('-'.join, axis=1)

varLst = ['EcoB1', 'EcoB2', 'EcoB3']
for var in varLst:
    df2[var] = df2[var].astype(int).astype(str).str.zfill(2)
dfT['EcoB'] = df2[varLst].agg('-'.join, axis=1)

dfT[dfT['Eco'] != dfT['EcoB']]
