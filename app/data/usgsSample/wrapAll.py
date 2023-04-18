from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd


siteNoFile=os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite=pd.read_csv(siteNoFile,dtype={'siteNo':str})
siteNoLst=dfSite['siteNo'].tolist()

dataName = 'dbAll'
codeLst = usgs.varC
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=codeLst+['00060','00065'],varF=gridMET.varLst,
    varG=gageII.varLstEx,edStr='2023-01-01')
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
# DF = dbBasin.DataFrameBasin(dataName)

# got some memory issue solve later