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
tabDMC = pd.read_csv(fileDMC)


specLst1 = tab['try_id'].unique().tolist()
specLst2 = tabDMC['AccSpeciesID'].unique().tolist()

specSet = list(set(specLst1).intersection(set(specLst2)))
spec = specSet[3]

tab1 = tab[tab['try_id'] == spec][['Latitude', 'Longitude', 'LFMC value']]
tab2 = tabDMC[tabDMC['AccSpeciesID'] == spec][['Latitude', 'Longitude', 'StdValue']]

s1 = tab1.groupby(['Latitude', 'Longitude']).nunique().index.values
s1