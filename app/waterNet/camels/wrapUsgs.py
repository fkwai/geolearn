from hydroDL.data import dbBasin, gageII, camels, gridMET, GLASS
import os
import pandas as pd
from hydroDL import kPath, utils


# all site
fileSiteAll = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst')
# fileSiteAll = os.path.join(kPath.dirData, 'USGS', 'inventory','siteSel','Q90')
siteNoAll = pd.read_csv(fileSiteAll, header=None, dtype=str)[0].tolist()

# camels sites
siteNoLst = camels.siteNoLst()
siteNoCamels = [x for x in siteNoAll if x in siteNoLst]


dataName = 'camelsK'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoCamels, varF=gridMET.varLst+GLASS.varLst, varG=gageII.varLstEx)

DF = dbBasin.DataFrameBasin(dataName)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
DF.saveSubset('WYall', sd='1982-01-01', ed='2018-12-31')

dataName = 'camelsK'
DF = dbBasin.DataFrameBasin(dataName)
DF.saveSubset('WY8095', sd='1980-10-01', ed='1995-09-30')
DF.saveSubset('WY9510', sd='1995-10-01', ed='2010-09-30')