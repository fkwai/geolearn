from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
import json

label = 'B200'
codeLst = usgs.varC.copy()

# dictVar=gageII.getVariableDict()
# dictVar['Regions']

for code in codeLst:
    dataName = '{}-{}'.format(code, label)
    DF = dbBasin.DataFrameBasin(dataName)
    dfR = gageII.readData(varLst=['HUC8_SITE'], siteNoLst=DF.siteNoLst)
    dfR['HUC2'] = dfR['HUC8_SITE'].astype(str).str.zfill(8).str[:2].astype(int)
    b = 0
    for h in dfR['HUC2'].unique():
        siteSubset = dfR[dfR['HUC2'] == h].index.tolist()
        sy = DF.t[0].astype(object).year
        ey = DF.t[-1].astype(object).year
        yrAry = np.array(range(sy, ey))
        yrIn = yrAry[yrAry % 5 == b]
        t1 = dbBasin.func.pickByYear(DF.t, yrIn)
        t2 = dbBasin.func.pickByYear(DF.t, yrIn, pick=False)
        DF.createSubset('pkYr5b{}_HUC{:02d}'.format(b, h), 
                        siteNoLst=siteSubset, dateLst=t1)
        DF.createSubset('rmYr5b{}_HUC{:02d}'.format(b, h), 
                        siteNoLst=siteSubset, dateLst=t2)
