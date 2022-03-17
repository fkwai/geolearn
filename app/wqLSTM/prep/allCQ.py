import scipy
import time
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL.master import basins
from hydroDL.data import gageII, usgs, gridMET
from hydroDL import kPath, utils
import os
import pandas as pd
import numpy as np
from hydroDL import kPath
from hydroDL.data import dbBasin, usgs

# create a dataFrame contains all C and Q

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()


# varG = ['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'DRAIN_SQKM']
# DF = dbBasin.DataFrameBasin.new(
#     'allCQ', siteNoLstAll, varF=['pr'], varQ=['00060'], varG=varG)

DF = dbBasin.DataFrameBasin('allCQ')