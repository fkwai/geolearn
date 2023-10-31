from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
import json

label = 'B200'
codeLst=usgs.varC.copy()

for code in codeLst:
    # subset    
    dataName = '{}-{}'.format(code, label)
    DF = dbBasin.DataFrameBasin(dataName)
    sy = DF.t[0].astype(object).year
    ey = DF.t[-1].astype(object).year
    yrAry = np.array(range(sy, ey))
    for k in range(5):
        yrIn = yrAry[yrAry % 5 == k]
        t1 = dbBasin.func.pickByYear(DF.t, yrIn)
        t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
        DF.createSubset('pkYr5b{}'.format(k), dateLst=t1)
        DF.createSubset('rmYr5b{}'.format(k), dateLst=t2)
    # before after 2015
    DF.saveSubset('B15', ed='2015-12-31')
    DF.saveSubset('A15', sd='2016-01-01')


# dictVar=gageII.getVariableDict()
# dictVar['Regions']
code='00915'
dataName = '{}-{}'.format(code, label)
DF = dbBasin.DataFrameBasin(dataName)
dfR=gageII.readData(varLst=['HUC8_SITE'],siteNoLst=DF.siteNoLst)

dfR['HUC2']=dfR['HUC8_SITE'].astype(str).str.zfill(8).str[:2].astype(int)

import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'int_col': [1, 23, 456, 7890]
})

# Convert to 8-digit strings and extract first two digits
df['new_col'] = df['int_col'].astype(str).str.zfill(8).str[:2].astype(int)

print(df)
