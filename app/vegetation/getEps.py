import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
import numpy as np

DIR_VEG = r'/home/kuai/work/VegetationWater/data/'
fileLFMC = os.path.join(DIR_VEG, 'LFMC-global.csv')
tabLFMC = pd.read_csv(fileLFMC)
filePV = os.path.join(DIR_VEG, 'PV-Bartlet2.csv')
tabPV = pd.read_csv(filePV)
fileFix = os.path.join(DIR_VEG, 'spec-LFMC-Bartlet-fix')
tabFix = pd.read_csv(fileFix)

# find useful rows
temp = tabLFMC[tabLFMC['Sampling date'] > 20150101]
tabData = temp[temp['Species collected'].isin(tabFix['spec1'])]



# add id
temp = tabData['ID'].str.split('_', 2, expand=True)
siteId = temp[0].str.cat(temp[1], sep='_')
siteIdLst = siteId.unique().tolist()
tabData['siteId'] = siteId

# add Bartlet spec
temp = tabFix.merge(tabPV, left_on='spec2', right_on='Species')
tabMerge = tabData.merge(temp, left_on='Species collected', right_on='spec1')

# add try id
fileSpec = os.path.join(DIR_VEG, 'spec-fix')
dfFix = pd.read_csv(fileSpec, index_col=0)
tabMerge = tabMerge.merge(dfFix, left_on='Species collected', right_index=True)


# map counts
tabSite = tabMerge[['siteId', 'Sitename', 'Latitude', 'Longitude']].drop_duplicates()
tabSpec = tabMerge[
    ['siteId', 'Sitename', 'Latitude', 'Longitude', 'Species']
].drop_duplicates()
cntSpec = tabMerge.groupby(['siteId'])['Species'].nunique().reset_index('siteId')
cntSample = tabMerge.groupby(['siteId'])['Species'].count().reset_index('siteId')
tabSite = tabSite.set_index('siteId')
lat = tabSite.loc[cntSpec['siteId']]['Latitude'].values
lon = tabSite.loc[cntSpec['siteId']]['Longitude'].values

# number of spec
s1=tabMerge.groupby(['Species'])['siteId'].nunique().sort_values(ascending=False)
s2=tabMerge.groupby(['Species'])['try_id'].count().sort_values(ascending=False)
df=pd.DataFrame(s1).join(s2)
df=df.merge(tabPV, left_on='Species', right_on='Species')
df[['Species','siteId','try_id','eps(MPa)']]

# map of all
extentUS = [-125, -65, 25, 50]
extentEU = [-0, 10, 40, 45]
extentGlobal = [-180, 180, -90, 90]

fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax = mapplot.mapPoint(fig, gs[0, 0], lat, lon, cntSample['Species'], extent=extentUS)
ax = mapplot.mapPoint(fig, gs[1, 0], lat, lon, cntSpec['Species'], extent=extentUS)
fig.show()

fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
ax = mapplot.mapPoint(fig, gs[0, 0], lat, lon, cntSample['Species'], extent=extentEU)
ax = mapplot.mapPoint(fig, gs[1, 0], lat, lon, cntSpec['Species'], extent=extentEU)
fig.show()

epsMean = tabMerge.groupby(['siteId'])['eps(MPa)'].mean().reset_index('siteId')
epsDiff = tabMerge.groupby(['siteId'])['eps(MPa)'].std().reset_index('siteId')
indM=cntSpec['Species']>1
indS=cntSpec['Species']==1


fig = plt.figure()
gs = gridspec.GridSpec(3, 1)
ax = mapplot.mapPoint(fig, gs[0, 0], lat[indS], lon[indS], epsMean[indS]['eps(MPa)'].values, extent=extentUS)
ax = mapplot.mapPoint(fig, gs[1, 0], lat[indM], lon[indM], epsMean[indM]['eps(MPa)'].values, extent=extentUS)
ax = mapplot.mapPoint(fig, gs[2, 0], lat[indM], lon[indM], epsDiff[indM]['eps(MPa)'].values, extent=extentUS)
fig.show()

fig = plt.figure()
gs = gridspec.GridSpec(3, 1)
ax = mapplot.mapPoint(fig, gs[0, 0], lat[indS], lon[indS], epsMean[indS]['eps(MPa)'].values, extent=extentEU)
ax = mapplot.mapPoint(fig, gs[1, 0], lat[indM], lon[indM], epsMean[indM]['eps(MPa)'].values, extent=extentEU)
ax = mapplot.mapPoint(fig, gs[2, 0], lat[indM], lon[indM], epsDiff[indM]['eps(MPa)'].values, extent=extentEU)
fig.show()
