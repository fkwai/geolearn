import os
import pandas as pd
import numpy as np
from hydroDL import kPath
from hydroDL.data import dbBasin


fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()


dataName = 'All1982'
sd = '1982-01-01'
ed = '2018-12-31'
freq = 'D'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLstAll, sdStr=sd, edStr=ed, freq=freq)
