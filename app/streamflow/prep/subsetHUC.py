import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np


dirEco = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
fileEco = os.path.join(dirEco, 'basinHUC')
dfEco = pd.read_csv(fileEco, dtype={'siteNo': str}).set_index('siteNo')
# dataName = 'Q90ref'
dataName = 'Q90'
dm = dbBasin.DataModelFull(dataName)

#
nameLst = list()
subsetLst = list()
regionLst = sorted(dfEco['HUC02'].unique().tolist())
for regionId in regionLst:
    temp = dfEco['HUC02'] == regionId
    ss = dfEco[temp].index.tolist()
    subset = list(set(dm.siteNoLst).intersection(ss))
    subsetLst.append(subset)
    subsetName = 'HUC{:02d}'.format(regionId)
    nameLst.append(subsetName)
dm.saveSubset(nameLst, subsetLst)
