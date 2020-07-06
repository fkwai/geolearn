from hydroDL import kPath, utils
from hydroDL.app import waterQuality, wqRela
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from astropy.timeseries import LombScargle
import scipy.signal as signal
import matplotlib.gridspec as gridspec


# pick out sites that are have relative large number of observations
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfAll = pd.read_csv(os.path.join(dirInv, 'codeCount.csv'),
                    dtype={'siteNo': str}).set_index('siteNo')

# pick some sites
# codeLst = ['00915', '00940', '00955','00300']
codeLst = ['00660', '00600']
startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2019, 12, 31)
siteNo = '07060710'
dfC, dfCF = usgs.readSample(siteNo, codeLst=codeLst,
                            startDate=startDate, flag=True)
dfQ = usgs.readStreamflow(siteNo, startDate=startDate)


fig, axes = plt.subplots(len(codeLst), 1)
for k, code in enumerate(codeLst):
    flagLst = ['<', 'E']
    axes[k].plot(dfC[code], '*', label='others')
    for flag in flagLst:
        axes[k].plot(dfC[code][dfCF[code+'_cd'] == flag], '*', label=flag)
    shortName = usgs.codePdf.loc[code]['shortName']
    title = '{} {} {}'.format(siteNo,shortName, code)
    axes[k].set_title(title)
    axes[k].legend()
fig.show()

fig, axes = plt.subplots(len(codeLst), 1)
# axplot.plotTS(axes[0],  dfQ.index.values,
#               dfQ['00060_00003'].values, styLst='-')
# axes[0].set_title(' {} streamflow'.format(siteNo))
for k, code in enumerate(codeLst):
    vc = dfC[code].values.copy()
    vf = dfCF[code+'_cd'].values
    vcf = dfC[code].values.copy()
    vcf[(vf == 'x') | (vf == 'X')] = np.nan
    data = [vc, vcf]
    axplot.plotTS(axes[k],  dfC.index.values, data, styLst='**', cLst='rk')
    shortName = usgs.codePdf.loc[code]['shortName']
    title = '{} {}'.format(shortName, code)
    axes[k+1].set_title(title)
fig.show()
