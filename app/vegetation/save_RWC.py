import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
import numpy as np
from hydroDL import kPath

DIR_VEG = os.path.join(kPath.dirVeg)

# load LFMC
fileLFMC = os.path.join(DIR_VEG, 'LFMC-global.csv')
tabLFMC = pd.read_csv(fileLFMC)

# date > 2015
tabData = tabLFMC
tabData = tabLFMC[tabLFMC['Sampling date'] > 20150101]
tabLFMC['Sampling date'].max()


# add site id
temp = tabData['ID'].str.split('_', 2, expand=True)
siteId = temp[0].str.cat(temp[1], sep='_')
siteIdLst = siteId.unique().tolist()
tabData['siteId'] = siteId

# add spec id
fileSpec = os.path.join(DIR_VEG, 'spec-fix')
dfFix = pd.read_csv(fileSpec, index_col=0)
tab = tabData.merge(dfFix, left_on='Species collected', right_index=True)
tab[tab['Sitename'] == 'Shoshone Basin']
len(tab)


# with open('temp', 'w') as fp:
#     for item in sorted(tab['try_id'].unique().tolist()):
#         fp.write('{}, '.format(item))


# load DMC
fileDMC = os.path.join(DIR_VEG, 'TRY', 'DMC.csv')
tabDMC_temp = pd.read_csv(fileDMC)
meanDMC = tabDMC_temp.groupby(['AccSpeciesID'])['StdValue'].mean()
minDMC = tabDMC_temp.groupby(['AccSpeciesID'])['StdValue'].quantile(0.1)
stdDMC = tabDMC_temp.groupby(['AccSpeciesID'])['StdValue'].std()
minDMC = minDMC.rename('DMC')
tab = tab.merge(minDMC, left_on='try_id', right_on='AccSpeciesID')
tab['LFMC'] = tab['LFMC value'] / 100
tab['date'] = pd.to_datetime(tab['Sampling date'], format='%Y%m%d')
tab['RWC'] = tab['DMC'] * tab['LFMC'] / (1 - tab['DMC'])


# extract sites
tabPlot = tab[['siteId', 'date', 'try_id', 'try_spec', 'LFMC', 'RWC', 'DMC']]
tabSite = tab[
    ['siteId', 'Sitename', 'State/Region', 'Latitude', 'Longitude']
].drop_duplicates()
cntSample = tabPlot.groupby(['siteId'])['try_id'].count().reset_index('siteId')
tabSite = tabSite.merge(cntSample, left_on='siteId', right_on='siteId')
# save sites
tabS = tabSite[['siteId', 'Latitude', 'Longitude']]
tabS.to_csv(os.path.join(DIR_VEG, 'rwc_sites.csv'), index=False)


tabOut = tab[
    ['RWC', 'siteId', 'date', 'DMC', 'try_id', 'Elevation(m.a.s.l)', 'Slope(%)']
]
tabOut = tabOut.rename(
    columns={
        'RWC': 'percent',
        'siteId': 'site',
        'Elevation(m.a.s.l)': 'elevation',
        'Slope(%)': 'slope',
    }
)
tabOut = tabOut.replace({'slope': {'< 10': 10.0}})
tabOut['slope'] = tabOut['slope'].astype(float)

tabOut.to_csv(os.path.join(DIR_VEG, 'rwc.csv'), index=False)


tabOut = tab[['RWC', 'siteId', 'date', 'DMC', 'try_id', 'Elevation(m.a.s.l)']]
tabOut = tabOut.rename(
    columns={'RWC': 'percent', 'siteId': 'site', 'Elevation(m.a.s.l)': 'elevation'}
)
# tabOut = tabOut.replace({'slope': {'< 10': 10.0}})
# tabOut['slope'] = tabOut['slope'].astype(float)

tabOut.to_csv(os.path.join(DIR_VEG, 'rwc.csv'), index=False)
