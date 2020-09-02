from hydroDL import kPath, utils
from hydroDL.data import ntn
import pandas as pd
import numpy as np
import os
import time
import importlib

# save ntn data to csv files
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
tabData = ntn.readDataRaw()
ntnIdLst = tabData['siteID'].unique().tolist()
varLst = ntn.varLst

folder1 = os.path.join(dirNTN, 'csv', 'weekly')
folder2 = os.path.join(dirNTN, 'csv', 'weeklyRaw')

for kk, ntnId in enumerate(ntnIdLst):
    df1 = pd.read_csv(os.path.join(folder1, ntnId), index_col='date')
    df2 = pd.read_csv(os.path.join(folder2, ntnId), index_col='date')
    df = df1-df2
    data = df.dropna(how='all').values
    print('{} {} {}'.format(kk, ntnId, np.nanmean(data)))
