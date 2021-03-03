from hydroDL.app import waterQuality
from hydroDL.data import usgs
import numpy as np
import pandas as pd
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt


siteNo = '09163500'
varC = ['00660', '00618']

sd = np.datetime64('1979-01-01')
ed = np.datetime64('2019-12-31')

df = waterQuality.readSiteTS(
    siteNo, varLst=['00060']+varC)

dfC, dfCF = usgs.readSample(siteNo, codeLst=varC, startDate=sd, flag=2)

#
fig, axes = plt.subplots(2, 1)
for k, code in enumerate(varC):
    v = dfC[code].values
    f = dfCF[code+'_cd'].values
    t = dfC.index.values
    indF = np.where(f == 1)[0]
    axplot.plotTS(axes[k], t, v, cLst='r', styLst=['-*'])
    axplot.plotTS(axes[k], t[indF], v[indF], cLst='b', styLst='*')
fig.show()
