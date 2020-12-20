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
fileEco = os.path.join(dirEco, 'basinEco')
dfEco = pd.read_csv(fileEco, dtype={'siteNo': str}).set_index('siteNo')


dataName = 'Q90'
dm = dbBasin.DataModelFull(dataName)

siteNoLst = dm.siteNoLst
regionLst = sorted(dfEco['region'].unique().tolist())
regionId = 'A'

dfTemp = dfEco.loc[siteNoLst]

dfTemp['region'].value_counts().sort_index()
