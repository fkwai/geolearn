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
meanDMC = tabDMC.groupby(['AccSpeciesID'])['StdValue'].mean()
stdDMC = tabDMC.groupby(['AccSpeciesID'])['StdValue'].std()
stdDMC = stdDMC.rename('DMC_std')


tabTemp = tab.copy()
q = 0.05
minDMC = tabDMC.groupby(['AccSpeciesID'])['StdValue'].quantile(q)
minDMC = minDMC.rename('DMC')
meanDMC = meanDMC.rename('DMC_mean')


tabTemp = tabTemp.merge(minDMC, left_on='try_id', right_on='AccSpeciesID')
tabTemp = tabTemp.merge(meanDMC, left_on='try_id', right_on='AccSpeciesID')

tabTemp['LFMC'] = tabTemp['LFMC value'] / 100
tabTemp['date'] = pd.to_datetime(tabTemp['Sampling date'], format='%Y%m%d')
tabTemp['RWC'] = tabTemp['DMC'] * tabTemp['LFMC'] / (1 - tabTemp['DMC'])
tabTemp['RWC_mean'] = tabTemp['DMC_mean'] * tabTemp['LFMC'] / (1 - tabTemp['DMC_mean'])


fig, ax = plt.subplots(1, 1)
ax.plot(tabTemp['RWC'], tabTemp['DMC'], 'b*')
ax.plot([1, 1], [0, tabTemp['DMC'].max()], '-r')
r = len(tabTemp[tabTemp['RWC'] > 1]) / len(tabTemp)
ax.set_xlabel('RWC')
ax.set_ylabel('DMC')
ax.set_title('quantile = {}, RWC>1: {:.2f}'.format(q, r))
ax.legend()
fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(tabTemp['RWC'], tabTemp['RWC_mean'], '*')
ax.plot([0, tabTemp['RWC'], tabTemp['RWC_mean']], [0, tabTemp['RWC_mean'].max()], '-r')
r = len(tabTemp[tabTemp['RWC'] > 1]) / len(tabTemp)
ax.set_xlabel('RWC')
ax.set_ylabel('DMC')

ax.legend()
fig.show()
