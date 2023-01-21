import os
import pandas as pd
import numpy as np
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

DIR_VEG = r'/home/kuai/work/VegetationWater/data/'
filePsi = os.path.join(DIR_VEG, 'psi_fluxnet.csv')
tabPsi = pd.read_csv(filePsi)
tabPsi['t']=pd.to_datetime(tabPsi['TIMESTAMP'])

t=tabPsi['t'].values
t0=np.datetime64('2015-01-01')
len(np.where(t>t0)[0])

tabPsi.columns

siteNameLst=tabPsi['site_name'].unique()

for siteName in siteNameLst:
    tab= tabPsi[tabPsi['site_name'] == siteName]
    spec=tab['pl_species'].unique()
    print(len(spec))


for siteName in siteNameLst:
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))

    tab= tabPsi[tabPsi['site_name'] == siteName]
    specLst=tab['pl_species'].unique()

    for spec in specLst:
        temp=tab[tab['pl_species']==spec]
        axP.plot(temp['t'],temp['psi'],'*')    
    axP.legend()
    figP.show()
    figP.savefig(siteName)
