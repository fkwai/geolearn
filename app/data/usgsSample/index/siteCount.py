import pandas as pd
from hydroDL.data import usgs
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath
import os
from hydroDL.app.waterQuality import cqType
import importlib

import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

tabCount=pd.read_csv(os.path.join(kPath.dirUsgs,  'index', 'sampleCount_a79_v20.csv'),
                   dtype={'siteNo':str}).set_index('siteNo')

codeGroup = [
    ['00010', '00300'],
    ['00915', '00925', '00930', '00955'],
    ['00600', '00605', '00618', '00660', '00665', '00681', '71846'],
    ['00095', '00400', '00405', '00935', '00940', '00945', '80154']
]

codeStr = usgs.codePdf.loc[codeLst[k]]['shortName']


codeG=codeGroup[3]
n=len(tabCount)
lb=100
ub=1000
fig,ax=plt.subplots(1,1)
for code in codeG:
    v=np.sort(tabCount[code].values)[::-1]
    v[v>ub]=ub
    vv=v[v>lb]
    codeStr = usgs.codePdf.loc[code]['shortName']
    ax.plot(np.arange(len(vv)),vv)


ax.legend()
fig.show()

