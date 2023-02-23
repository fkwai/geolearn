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
medianDMC1 = tabDMC.groupby(['AccSpeciesID','Latitude', 'Longitude'])['StdValue'].median()


meanDMC2 = tabDMC.groupby(['AccSpeciesID'])['StdValue'].mean()
stdDMC2 = tabDMC.groupby(['AccSpeciesID'])['StdValue'].std()

fig, ax = plt.subplots(1, 1)
ax.plot(meanDMC1, medianDMC1, 'r*',label='per spec per site')
ax.plot(meanDMC2, stdDMC2, 'b*',label='per spec')
ax.set_xlabel('mean of DMC')
ax.set_ylabel('std of DMC')
fig.show()

meanDMC = meanDMC.rename('DMC')
stdDMC = stdDMC.rename('DMC_std')

tabDMC1 = pd.merge(meanDMC1, stdDMC1, right_index=True, left_index=True)

20484
tabDMC[tabDMC['AccSpeciesID']==20484].to_csv('temop')