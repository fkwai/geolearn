from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
import json

saveName = 'siteNoLst_79_23'
saveFile = os.path.join(kPath.dirUsgs, 'siteSel', saveName)
with open(saveFile, 'r') as f:
    dictSite = json.load(f)


code = '00955'
dataName = 'test'
siteNoLst = dictSite[code+'-B200'][:2]
code = '00955'
# dataName = '00955-B200'
# siteNoLst = dictSite[code+'-B200']
DF = dbBasin.DataFrameBasin.new(
    dataName,
    siteNoLst,
    varC=[code],
    varQ=usgs.varQ,
    varF=gridMET.varLst + GLASS.varLst,
    varG=gageII.varLstEx,
    sdStr='1979-01-01',
    edStr='2023-12-31',
)

DF = dbBasin.DataFrameBasin(dataName)
DF.saveSubset('WY_82_09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WY_09_18', sd='2009-10-01', ed='2018-12-31')