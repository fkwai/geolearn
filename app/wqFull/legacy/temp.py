import numpy as np
import pandas as pd
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn, transform, GLASS
from hydroDL.app import waterQuality
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot

siteNo = '10343500'
dfV = GLASS.readBasin(siteNo)

varF = gridMET.varLst+ntn.varLst+['distNTN']
varC = usgs.varC
varQ = usgs.varQ
varG = gageII.lstWaterQuality
varLst = varQ+varC+varF
df = waterQuality.readSiteTS(siteNo, varLst=varLst, freq='D')
df = df.join(dfV)

pVar = ['00915', 'NPP', '00060']
fig, axes = plt.subplots(len(pVar), 1)
for k, var in enumerate(pVar):
    axplot.plotTS(axes[k], df.index, df[var])
fig.show()


# interpolation of R
var = 'NPP'
sdStr = '1982-01-01'
edStr = '2018-12-31'
tR = pd.date_range(np.datetime64(sdStr), np.datetime64(edStr))
dfVP = pd.DataFrame({'date': tR}).set_index('date').join(df[var])
dfVP = dfVP.interpolate(method='cubicspline')
fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, dfV.index, dfV[var], styLst='*', cLst='r')
axplot.plotTS(ax, dfVP.index, dfVP[var], styLst='-', cLst='b')
fig.show()

fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()
ax.plot(df.index, df['00915'], '*r')
ax2.plot(dfVP.index, dfVP['NPP'],'-b')
fig.show()
