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
td = pd.date_range(sd, ed)

df = waterQuality.readSiteTS(
    siteNo, varLst=['00060']+varC)

dfC, dfCF = usgs.readSample(siteNo, codeLst=varC, startDate=sd, flag=2)
dfOut = usgs.removeFlag(dfC, dfCF)

#
fig, axes = plt.subplots(2, 1)
for k, code in enumerate(varC):
    v = dfC[code].values
    # f = dfCF[code+'_cd'].values
    f = dfOut[code].values
    t = dfC.index.values
    indF = np.where(f == 1)[0]
    axplot.plotTS(axes[k], t, v, cLst='r', styLst=['*'])
    # axplot.plotTS(axes[k], t[indF], v[indF], cLst='b', styLst='*')
    axplot.plotTS(axes[k], t, f, cLst='b', styLst=['*'])
fig.show()

# temp code - waterQuality.readSiteTS
flag = True
dfD = pd.DataFrame({'date': td}).set_index('date')
if flag:
    dfC, dfCF = usgs.readSample(siteNo, codeLst=varC, startDate=sd, flag=2)
    dfDF = pd.DataFrame({'date': td}).set_index('date')
    dfDF = dfDF.join(dfCF)
else:
    dfC = usgs.readSample(siteNo, codeLst=varC, startDate=sd, flag=flag)
dfD = dfD.join(dfC)
dfD.resample('W-TUE').mean()
dfDF.resample('W-TUE').max()


#

