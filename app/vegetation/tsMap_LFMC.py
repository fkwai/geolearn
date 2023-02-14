import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot


DIR_VEG = r'/home/kuai/work/VegetationWater/data/'
fileLFMC = os.path.join(DIR_VEG, 'LFMC-global.csv')
tabLFMC = pd.read_csv(fileLFMC)

# date
tabData = tabLFMC
tabData = tabLFMC[tabLFMC['Sampling date'] > 20150101]


# add id
temp = tabData['ID'].str.split('_', 2, expand=True)
siteId = temp[0].str.cat(temp[1], sep='_')
siteIdLst = siteId.unique().tolist()
tabData['siteId'] = siteId
# add spec
fileSpec = os.path.join(DIR_VEG, 'spec-fix')
dfFix = pd.read_csv(fileSpec, index_col=0)
tab = tabData.merge(dfFix, left_on='Species collected', right_index=True)

# load DMC
fileDMC = os.path.join(DIR_VEG, 'TRY', 'DMC.csv')
tabDMC_temp = pd.read_csv(fileDMC)
meanDMC = tabDMC_temp.groupby(['AccSpeciesID'])['StdValue'].mean()
stdDMC = tabDMC_temp.groupby(['AccSpeciesID'])['StdValue'].std()
meanDMC = meanDMC.rename('DMC')
stdDMC = stdDMC.rename('DMC_std')
tabDMC = pd.merge(meanDMC, stdDMC, right_index=True, left_index=True)

fig, ax = plt.subplots(1, 1)
ax.plot(tabDMC['DMC'], tabDMC['DMC_std'], '*')
ax.plot([0, 0.6], [0, 0.2], '-k')
fig.show()

tab = tab.merge(tabDMC, left_on='try_id', right_on='AccSpeciesID')
tab['LFMC'] = tab['LFMC value']/100
tab['date'] = pd.to_datetime(tab['Sampling date'], format='%Y%m%d')
tab['RWC'] = tab['DMC']*tab['LFMC']/(1-tab['DMC'])

tabPlot = tab[['siteId', 'date', 'try_id', 'try_spec', 'LFMC', 'RWC', 'DMC']]
tabSite = tab[['siteId', 'Latitude', 'Longitude']].drop_duplicates()
cntSample = tabPlot.groupby(['siteId'])['try_id'].count().reset_index('siteId')
tabSite = tabSite.merge(cntSample, left_on='siteId', right_on='siteId')

lat = tabSite['Latitude'].values
lon = tabSite['Longitude'].values
extentUS = [-125, -65, 25, 50]
extentEU = [-5, 15, 40, 45]
extentGlobal = [-180, 180, -90, 90]


fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax = mapplot.mapPoint(fig, gs[0, 0], lat, lon, tabSite['try_id'], extent=extentUS)
ax = mapplot.mapPoint(fig, gs[1, 0], lat, lon, tabSite['try_id'], extent=extentEU)
fig.show()

# plot time series
siteId = 'C6_18'
tabData = tabPlot[tabPlot['siteId'] == siteId]
specLst = tabData['try_spec'].unique().tolist()

cLst = 'rgbkmcy'
fig, axes = plt.subplots(2, 1)
for spec in specLst:
    temp = tabData[tabData['try_spec'] == spec]
    axes[0].plot(temp['date'], temp['LFMC'], '-*', label=spec)
    axes[1].plot(
        temp['date'], temp['RWC'], '-*', label='{:.2f}'.format(temp.iloc[0]['DMC'])
    )
axes[0].legend()
axes[1].legend()
fig.show()


import numpy as np


def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, tabSite['try_id'], extent=extentGlobal)    
    figP, axP = plt.subplots(2, 1)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteId = tabSite.iloc[iP]['siteId']
    tabData = tabPlot[tabPlot['siteId'] == siteId]
    specLst = tabData['try_spec'].unique().tolist()
    for spec in specLst:
        temp = tabData[tabData['try_spec'] == spec]
        axP[0].plot(temp['date'], temp['LFMC'], '-*', label=spec)
        axP[1].plot(
            temp['date'], temp['RWC'], '-*', label='{:.2f}'.format(temp.iloc[0]['DMC'])
        )
    axP[0].legend()
    axP[1].legend()


figplot.clickMap(funcM, funcP)

a=1
type(a) is int