
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
code = '00660'

siteNo = '01111500'
df = waterQuality.readSiteTS(siteNo, [code], freq='D').dropna()
dfC, dfCF = usgs.readSample(siteNo, codeLst=[code], flag=2)
dfC = dfC.resample('W-TUE').mean()
dfCF = dfCF.fillna(0)
dfCFW = dfCF.resample('W-TUE').mean()
dfCFW = dfCFW.fillna(0)
dfCFW[dfCFW != 0] = 1
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
t = dfC.index
v = dfC[code].values
flag = dfCFW[code+'_cd'].values
ax.plot(t[flag == 0], v[flag == 0], 'r*')
ax.plot(t[flag != 0], v[flag != 0], 'k*')
fig.show()
