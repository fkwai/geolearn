import pandas as pd
from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import json
import os
from hydroDL import kPath
import numpy as np
from hydroDL.post import axplot, figplot

saveName = 'siteNoLst_79_23'
saveFile = os.path.join(kPath.dirUsgs, 'siteSel', saveName)
with open(saveFile, 'r') as f:
    dictSite=json.load(f)
siteNoLst=dictSite['any-B200']

dataName = 'any-B200'
DF = dbBasin.DataFrameBasin.new(
    dataName,
    siteNoLst,
    varC=usgs.varC,
    varF=gridMET.varLst,
    varG=gageII.varLstEx,
    sdStr='1979-01-01',
    edStr='2022-12-31'
)