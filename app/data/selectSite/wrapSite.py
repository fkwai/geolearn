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

label = 'B200'
code = '00915'

# DF for single codes
for code in ['00915', '00955', '00618']:
    siteName = '{}-{}'.format(code, label)
    dataName = '{}-{}'.format(code, label)
    siteNoLst = dictSite[siteName]
    DF = dbBasin.DataFrameBasin.new(
        dataName,
        siteNoLst,
        varC=[code],
        varQ=usgs.varQ,
        varF=gridMET.varLst,
        varG=gageII.varLstEx,
        sdStr='1979-01-01',
        edStr='2023-01-01',
    )
