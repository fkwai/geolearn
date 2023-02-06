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
specLFMC = tabData['Species collected'].unique().tolist()

# broke up mixed
specLst = list()
for k, spec in enumerate(specLFMC):
    if not pd.isnull(spec) and not spec[:7] == 'Unknown':
        if ',' in spec:
            tempLst = spec.split(',')
            for temp in tempLst:
                if temp[0] == ' ':
                    temp = temp[1:]
                specLst.append(temp)
        else:
            specLst.append(spec)
len(specLst)
specLst1 = sorted(set(specLst))

# Bartlet species
filePV = os.path.join(DIR_VEG, 'PV-Bartlet2.csv')
tabPV = pd.read_csv(filePV)
specLst = tabPV.loc[~tabPV['eps(MPa)'].isna()]['Species'].to_list()
specLst2 = sorted(set(specLst))

# potential fix
from difflib import SequenceMatcher
import numpy as np



mat = np.zeros([len(specLst1),len(specLst2)])
for j,spec1 in enumerate(specLst1):
    for i, spec2 in enumerate(specLst2):
        temp = spec1.split(' ')
        if len(temp) == 1:
            temp.append('sp.')
        elif temp[1] == 'spp.':
                temp[1] = 'sp.'
        ss=temp[0] + ' ' + temp[1]           
        mat[j,i]=SequenceMatcher(None, ss, spec2).ratio()

dictSpec=dict()
mat2=np.max(mat,axis=1)
ind2=np.argmax(mat,axis=1)
ind=np.argsort(mat2)[::-1]
dictSpec['spec1']=[specLst1[k] for k in ind]
dictSpec['spec2']=[specLst2[ind2[k]] for k in ind]
dictSpec['v']=mat2[ind]
outFile = os.path.join(DIR_VEG, 'spec-LFMC-Bartlet')
df=pd.DataFrame.from_dict(dictSpec)
df.to_csv(outFile)


# add id
temp = tabData['ID'].str.split('_', 2, expand=True)
siteId = temp[0].str.cat(temp[1], sep='_')
siteIdLst = siteId.unique().tolist()
tabData['siteId'] = siteId
# add spec
fileSpec = os.path.join(DIR_VEG, 'spec-fix')
dfFix = pd.read_csv(fileSpec, index_col=0)
tab = tabData.merge(dfFix, left_on='Species collected', right_index=True)

# print availble in barLet
tabTemp = tab.loc[tab['Barlet']]
filePV = os.path.join(DIR_VEG, 'PV-Bartlet2.csv')
tabPV = pd.read_csv(filePV)
