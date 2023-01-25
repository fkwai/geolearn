
from hydroDL.data import dbBasin, usgs, gageII, gridMET
import os
from hydroDL import kPath
import numpy as np
import pandas as pd


siteNoFile=os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite=pd.read_csv(siteNoFile,dtype={'siteNo':str})
siteNoLst=dfSite['siteNo'].tolist()

dataName = 'dbAll'
codeLst = usgs.varC+usgs.codeIso
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=codeLst, varF = gridMET.varLst,
    varG=gageII.varLstEx,edStr='2022-01-01')
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
DF = dbBasin.DataFrameBasin(dataName)
