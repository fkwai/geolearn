import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np

caseLst = ['080305', '090301', '090303',
           '090401', '090402', '100105', '100204']

dirEco = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
fileEco = os.path.join(dirEco, 'basinCode')
dfEco = pd.read_csv(fileEco, dtype={'siteNo': str}).set_index('siteNo')


dataName = 'Q90ref'
dm = dbBasin.DataModelFull(dataName)

nameLst = list()
subsetLst = list()

for case in caseLst:
    lev0 = int(case[0:2])
    lev1 = int(case[2:4])
    lev2 = int(case[4:6])
    temp0 = dfEco['code0'] == lev0
    temp1 = (dfEco['code0'] == lev0) & (dfEco['code1'] == lev1)
    temp2 = (dfEco['code0'] == lev0) & (
        dfEco['code1'] == lev1) & (dfEco['code2'] == lev2)
    ss = dfEco[temp0].index.tolist()
    subsetLst.append(list(set(dm.siteNoLst).intersection(ss)))
    nameLst.append(case[:2])
    ss = dfEco[temp1].index.tolist()
    subsetLst.append(list(set(dm.siteNoLst).intersection(ss)))
    nameLst.append(case[:4])
    ss = dfEco[temp2].index.tolist()
    subsetLst.append(list(set(dm.siteNoLst).intersection(ss)))
    nameLst.append(case[:6])
    dm.saveSubset(nameLst, subsetLst)
