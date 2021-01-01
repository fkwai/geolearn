import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np

# caseLst = ['080304',
#            '050301',
#            '080401',
#            '090203',
#            '080305',
#            '080203',
#            '080503',
#            '090402',
#            '080301',
#            '080107',
#            '080204',
#            '080402']
caseLst = ['0801', '0802', '0803', '0804', '0805', '0902', '0903', '0904']

dirEco = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
fileEco = os.path.join(dirEco, 'basinEcoB')
dfEco = pd.read_csv(fileEco, dtype={'siteNo': str}).set_index('siteNo')


dataName = 'Q90ref'
# dataName = 'Q90'
dm = dbBasin.DataModelFull(dataName)

nameLst = list()
subsetLst = list()

for case in caseLst:
    lev0 = int(case[0:2])
    lev1 = int(case[2:4])
    # lev2 = int(case[4:6])
    temp0 = dfEco['EcoB1'] == lev0
    temp1 = (dfEco['EcoB1'] == lev0) & (dfEco['EcoB2'] == lev1)
    # temp2 = (dfEco['EcoB1'] == lev0) & (
    #     dfEco['EcoB2'] == lev1) & (dfEco['EcoB3'] == lev2)
    ss = dfEco[temp0].index.tolist()
    subsetLst.append(list(set(dm.siteNoLst).intersection(ss)))
    nameLst.append('EcoB'+case[:2])
    ss = dfEco[temp1].index.tolist()
    subsetLst.append(list(set(dm.siteNoLst).intersection(ss)))
    nameLst.append('EcoB'+case[:4])
    # ss = dfEco[temp2].index.tolist()
    # subsetLst.append(list(set(dm.siteNoLst).intersection(ss)))
    # nameLst.append('EcoB'+case[:6])
dm.saveSubset(nameLst, subsetLst)
