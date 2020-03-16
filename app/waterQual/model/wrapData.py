from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII
import pandas as pd
import numpy as np
import os
import time

# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

# case for all gages
# caseName = 'basinAll'
# # wqData = waterQuality.DataModelWQ.new(caseName, siteNoLstAll)
# wqData = waterQuality.DataModelWQ(caseName)
# ind1 = wqData.indByRatio(0.8)
# ind2 = wqData.indByRatio(0.2, first=False)
# wqData.saveSubset(['first80', 'last20'], [ind1, ind2])

# # select referenced basins
# tabSel = gageII.readData(
#     varLst=['CLASS'], siteNoLst=siteNoLstAll)
# tabSel = gageII.updateCode(tabSel)
# siteNoLst = tabSel[tabSel['CLASS'] == 1].index.tolist()
# wqData = waterQuality.DataModelWQ.new('basinRef', siteNoLst)
# ind1 = wqData.indByRatio(0.8)
# ind2 = wqData.indByRatio(0.2, first=False)
# wqData.saveSubset(['first80', 'last20'], [ind1, ind2])

# case for all gages
caseName = 'temp10'
wqData = waterQuality.DataModelWQ.new(caseName, siteNoLstAll[:10])
# wqData = waterQuality.DataModelWQ(caseName)
ind1 = wqData.indByRatio(0.8)
ind2 = wqData.indByRatio(0.2, first=False)
wqData.saveSubset(['first80', 'last20'], [ind1, ind2])
