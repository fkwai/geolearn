from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time

# siteNoLst = ['0143400680', '01434021', '01434025']
siteNo = '01434025'
code = '00955'
dfC = waterQuality.readSiteY(siteNo, [code])

# convert to weekly
t = pd.date_range(start='1979-01-01', end='2019-12-30', freq='W-TUE')
t1 = pd.date_range(start='1979-01-02', end='2019-12-30', freq='W-TUE')
t2 = pd.date_range(start='1979-01-02', end='2019-12-30', freq='W-MON')

offset = pd.offsets.timedelta(days=-6)
dfW = dfC.resample('W-MON', loffset=offset).mean()

fig, ax = plt.subplots(1, 1)
ax.plot(dfC,'*')
# ax.plot(dfW,'*')
fig.show()