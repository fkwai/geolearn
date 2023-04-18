import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot


DIR_VEG = r'/home/kuai/work/VegetationWater/data/'

# load DMC
fileDMC = os.path.join(DIR_VEG, 'TRY', 'DMC.csv')
tabDMC = pd.read_csv(fileDMC)

# add sites
tabSite=tabDMC.groupby(['Latitude', 'Longitude']).ngroup()


meanDMC1 = tabDMC.groupby(['AccSpeciesID','Latitude', 'Longitude'])['StdValue'].mean()
stdDMC1 = tabDMC.groupby(['AccSpeciesID','Latitude', 'Longitude'])['StdValue'].std()
minDMC1 = tabDMC.groupby(['AccSpeciesID','Latitude', 'Longitude'])['StdValue'].quantile(0.5)



aa=tabDMC.groupby(['AccSpeciesID','Latitude', 'Longitude']).describe()['StdValue']
aa.to_csv('temp')

meanDMC2 = tabDMC.groupby(['AccSpeciesID'])['StdValue'].mean()
minDMC2 = tabDMC.groupby(['AccSpeciesID'])['StdValue'].quantile(0.5)

stdDMC2 = tabDMC.groupby(['AccSpeciesID'])['StdValue'].std()

fig, ax = plt.subplots(1, 1)
ax.plot(meanDMC1, minDMC1, 'r*',label='per spec per site')
ax.plot(meanDMC2, minDMC2, 'b*',label='per spec')
ax.set_xlabel('mean of DMC')
ax.set_ylabel('std of DMC')
ax.legend()
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(tabDMC['StdValue'], tabDMC['Latitude'], 'r*',label='per spec per site')
ax.set_xlabel('mean of DMC')
ax.set_ylabel('std of DMC')
ax.legend()
fig.show()


a = tabDMC.groupby(['AccSpeciesID'])['StdValue'].std()
b= tabDMC.groupby(['AccSpeciesID','Latitude', 'Longitude'])['StdValue'].agg(['mean','std'])
b['tryid']=b.index.get_level_values(0)

c=b.groupby(['tryid']).std()['mean']
d=pd.merge(a,c,right_index=True,left_index=True)

fig, ax = plt.subplots(1, 1)
ax.plot(d['StdValue'],d['mean'],'*',label='a species')
ax.set_xlabel('std of all')
ax.set_ylabel('std of mean of each site')
ax.plot([0,0.2],[0,0.2],'k-')
ax.legend()
fig.show()

dd=pd.merge(b,c,left_on='tryid',right_on='tryid')
fig, ax = plt.subplots(1, 1)
ax.plot(dd['std'],dd['mean_y'],'*',label='per species per site')
ax.set_xlabel('std within site')
ax.set_ylabel('std between site')
ax.plot([0,0.2],[0,0.2],'k-')
ax.legend()
fig.show()

(dd['std']>dd['mean_y']).sum()

import numpy as np
np.corrcoef(d['StdValue'],d['mean'])