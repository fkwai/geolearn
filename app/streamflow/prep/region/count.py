import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np


dirCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
fileCode = os.path.join(dirCode, 'basinEco')
dfCode = pd.read_csv(fileCode, dtype={'siteNo': str}).set_index('siteNo')

dfCode['code0'] = dfCode['code0'].astype(int).astype(str).str.zfill(2)
dfCode['code1'] = dfCode['code1'].astype(int).astype(str).str.zfill(2)
dfCode['code2'] = dfCode['code2'].astype(int).astype(str).str.zfill(2)
dfCode['comb'] = dfCode[['code0', 'code1', 'code2']].agg('-'.join, axis=1)

# dataName = 'Q90ref'
dataName = 'Q90'
dm = dbBasin.DataModelFull(dataName)
siteNoLst = dm.siteNoLst
dfTemp = pd.DataFrame(index=siteNoLst, columns=[
    'code0', 'code1', 'code2', 'comb'])
dfTemp.update(dfCode)
combLst = dfTemp['comb'].value_counts().index.tolist()
dfCount = pd.DataFrame(index=combLst, columns=['lev0', 'lev1', 'lev2'])

for comb in combLst:
    code0, code1, code2 = comb.split('-')
    temp0 = dfTemp['code0'] == code0
    temp1 = (dfTemp['code0'] == code0) & (dfTemp['code1'] == code1)
    temp2 = (dfTemp['code0'] == code0) & (
        dfTemp['code1'] == code1) & (dfTemp['code2'] == code2)
    dfCount.at[comb, 'lev0'] = temp0.sum()
    dfCount.at[comb, 'lev1'] = temp1.sum()
    dfCount.at[comb, 'lev2'] = temp2.sum()

dfCount.to_csv('temp')