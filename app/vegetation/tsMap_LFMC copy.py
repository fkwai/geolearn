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
fileDMC = os.path.join(DIR_VEG ,'TRY' ,'DMC.csv')
tabDMC_temp = pd.read_csv(fileDMC)
meanDMC=tabDMC_temp.groupby(['AccSpeciesID'])['StdValue'].mean()
stdDMC=tabDMC_temp.groupby(['AccSpeciesID'])['StdValue'].std()
meanDMC=meanDMC.rename('DMC')
stdDMC=stdDMC.rename('DMC_std')
tabDMC = pd.merge(meanDMC,stdDMC,right_index=True, left_index=True)

fig,ax = plt.subplots(1,1)
ax.plot(tabDMC['DMC'],tabDMC['DMC_std'],'*')
ax.plot([0,0.6],[0,0.2],'-k')
fig.show()

tab = tab.merge(tabDMC, left_on='try_id', right_on='AccSpeciesID')
tab['LFMC'] =tab['LFMC value'] 
tab['RWC'] = tab['LFMC'] / tab['DMC']



# print availble try id
tabTemp=tab
# tabTemp = tab.loc[tab['Barlet']]
# tabTemp = tab.loc[tab['Anderegg']]
tabSite = tabTemp[['siteId', 'Sitename', 'Latitude', 'Longitude']].drop_duplicates()
tabSpec = tabTemp[
    ['siteId', 'Sitename', 'Latitude', 'Longitude', 'Species collected']
].drop_duplicates()

cntSpec = tabTemp.groupby(['siteId'])['try_id'].nunique().reset_index('siteId')
cntSample = tabTemp.groupby(['siteId'])['try_id'].count().reset_index('siteId')
tabSite = tabSite.set_index('siteId')

lat = tabSite.loc[cntSpec['siteId']]['Latitude'].values
lon = tabSite.loc[cntSpec['siteId']]['Longitude'].values

# map of all
extentUS = [-125, -65, 25, 50]
extentEU = [-5, 15, 40, 45]
extentGlobal = [-180, 180, -90, 90]

fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax = mapplot.mapPoint(fig, gs[0, 0], lat, lon, cntSample['try_id'], extent=extentUS)
ax = mapplot.mapPoint(
    fig, gs[1, 0], lat, lon, cntSpec['try_id'], vRange=[1, 2], extent=extentUS
)
fig.show()

fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax = mapplot.mapPoint(fig, gs[0, 0], lat, lon, cntSample['try_id'], extent=extentEU)
ax = mapplot.mapPoint(
    fig, gs[1, 0], lat, lon, cntSpec['try_id'], vRange=[1, 2], extent=extentEU
)
fig.show()
