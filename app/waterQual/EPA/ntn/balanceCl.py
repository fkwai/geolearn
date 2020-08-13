import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
tabG = gageII.readData(
    varLst=['NWIS_DRAIN_SQKM', 'BASIN_BOUNDARY_CONFIDENCE'], siteNoLst=siteNoLstAll)

# read NTN
dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')
fileData = os.path.join(dirNTN, 'NTN-All-w.csv')
fileSite = os.path.join(dirNTN, 'NTNsites.csv')
ntnData = pd.read_csv(fileData)
ntnSite = pd.read_csv(fileSite)
ntnData['siteID'] = ntnData['siteID'].apply(lambda x: x.upper())
ntnData = ntnData.replace(-9, np.nan)
ntnIdLst = ntnData['siteID'].unique().tolist()
crdNTN = pd.read_csv(os.path.join(dirNTN, 'crdNTN.csv'), index_col='siteid')
crdNTN = crdNTN.drop(['CO83', 'NC30', 'WI19'])
crdUSGS = pd.read_csv(os.path.join(
    dirNTN, 'crdUSGS.csv'), dtype={'STAID': str})
crdUSGS = crdUSGS.set_index('STAID')
t = pd.date_range(start='1979-01-01', end='2019-12-31', freq='W-TUE')
t = t[1:]

# varC = usgs.varC
varC = ['00940']
varNtn = ['Cl', 'subppt']
# siteNoLst = ['0422026250', '04232050', '0423205010']
siteNo = '04193500'
# siteNo = '01184000'
siteNoLstAll.index(siteNo)

# find NTN sites
usgsId = siteNo
x = crdUSGS.loc[usgsId]['x']
y = crdUSGS.loc[usgsId]['y']
dist = np.sqrt((x-crdNTN['x'])**2+(y-crdNTN['y'])**2)
dist = dist.drop(dist[dist > 500*1000].index)
data = np.full([len(t), len(varNtn)], np.nan)
distOut = np.full(len(t), np.nan)
idOut = np.full(len(t), np.nan, dtype=object)
while len(dist) > 0:
    ntnId = dist.idxmin()
    # temp = dictNTN[ntnId].values
    tab = ntnData[ntnData['siteID'] == ntnId]
    tab.index = pd.to_datetime(tab['dateoff'])
    out = pd.DataFrame(index=t)
    tol = pd.Timedelta(3, 'D')
    out = pd.merge_asof(left=out, right=tab, right_index=True,
                        left_index=True, direction='nearest', tolerance=tol)
    temp = out[varNtn].values
    matNan = np.isnan(data)
    indRow = np.unique(np.where(matNan)[0])
    data[matNan] = temp[matNan]
    idOut[indRow] = ntnId
    distOut[indRow] = dist[ntnId]
    dist = dist.drop(ntnId)
    indRow = np.unique(np.where(np.isnan(data))[0])
    if len(indRow) == 0:
        break
    # end of while
distOut[indRow] = np.nan
idOut[indRow] = np.nan
dfP = pd.DataFrame(index=t, columns=varNtn, data=data)
dfP['distNTN'] = distOut
dfP['idNTN'] = idOut
dfP.index.name = 'date'

# read C, Q, F
dfC = usgs.readSample(siteNo, codeLst=varC)
dfQ = usgs.readStreamflow(siteNo)
dfF = gridMET.readBasin(siteNo)
# convert to weekly
td = pd.date_range(start='1979-01-01', end='2019-12-30', freq='D')
df = pd.DataFrame({'date': td}).set_index('date')
df = df.join(dfC)
df = df.join(dfQ)
df = df.join(dfF)
df = df.rename(columns={'00060_00003': '00060'})
dfW = df.resample('W-TUE').mean()
dfW = dfW.join(dfP)
dfW = dfW.loc[t]

# weekly load
dfW['Q'] = dfW['00060']*60*60*24*7*(0.3048**3)  # m^3/week
area = tabG.loc[siteNo]['NWIS_DRAIN_SQKM']*200
conf = tabG.loc[siteNo]['BASIN_BOUNDARY_CONFIDENCE']
load1 = dfW['subppt']*area*dfW['Cl']  # kg
load2 = dfW['Q']*dfW['00940']/1000  # kg

# plot
fig, axes = plt.subplots(3, 1, figsize=(12, 6))
axes[0].plot(dfW['subppt'], 'r-', label='Prcp, NTN [mm]')
axes[0].plot(dfW['pr']*7, 'b-', label='Prcp, gridMET [mm]')
axes[0].plot(dfW['Q']/area, 'g-', label='Run off, USGS [mm]')
axes[1].plot(dfW['distNTN']/1000, '*',
             label='distance from NTN to USGS site [km]')
axes[2].plot(load1, 'r--*', label='Cl load input [kg]')
axes[2].plot(load2, 'b--*', label='Cl load output [kg]')
fig.show()

for ax in fig.axes:
    ax.legend()
fig.canvas.draw()

# plot
fig, axes = plt.subplots(3, 1, figsize=(12, 6))
axes[0].plot(dfW['subppt'], 'r-', label='Prcp, NTN [mm]')
axes[0].plot(dfW['pr']*7, 'b-', label='Prcp, gridMET [mm]')
ax02 = axes[0].twinx()
ax02.plot(dfW['Q']/area, 'g-', label='Run off, USGS [mm]')
axes[1].plot(dfW['distNTN']/1000, '*',
             label='distance from NTN to USGS site [km]')
axes[2].plot(load1, 'r--*', label='Cl load input [kg]')
# axes[2].plot(load2, 'b--*', label='Cl load output [kg]')
ax22 = axes[2].twinx()
ax22.plot(load2, 'b--*', label='Cl load output [kg]')
fig.show()

for ax in fig.axes:
    ax.legend()
fig.canvas.draw()


for ax in fig.axes:
    ax.set_xlim(np.datetime64('2015-01-01'), np.datetime64('2020-01-01'))
    ax.legend()
fig.canvas.draw()
