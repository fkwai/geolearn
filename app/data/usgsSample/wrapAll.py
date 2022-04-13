
from hydroDL.data import dbBasin, usgs, gageII
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
import os
from hydroDL import kPath
import numpy as np
import pandas as pd

# not work due to memory issue
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

dataName = 'dbAll_82_18'
codeLst = usgs.varC+usgs.codeIso
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=codeLst, varG=gageII.varLstEx)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
DF = dbBasin.DataFrameBasin(dataName)
