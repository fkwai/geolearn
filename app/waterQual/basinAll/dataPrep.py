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

tabSel = gageII.readData(
    varLst=['CLASS'], siteNoLst=siteNoLstAll)
tabSel = gageII.updateCode(tabSel)
siteNoLst = tabSel[tabSel['CLASS'] == 1].index.tolist()

# wqData = waterQuality.DataModelWQ.new('basinRef', siteNoLst)
wqData = waterQuality.DataModelWQ('basinRef', rmFlag=True)

indYr1 = waterQuality.indYr(wqData.info, yrLst=[1979, 2000])[0]
wqData.saveSubset('Y8090', indYr1)
indYr2 = waterQuality.indYr(wqData.info, yrLst=[2000, 2020])[0]
wqData.saveSubset('Y0010', indYr2)
