import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
import numpy as np

DIR_VEG = r'/home/kuai/work/VegetationWater/data/'

# load LFMC
fileLFMC = os.path.join(DIR_VEG, 'LFMC-global.csv')
tabLFMC = pd.read_csv(fileLFMC)

# date > 2015
tabLFMC = tabLFMC[tabLFMC['Sampling date'] > 20150101]

# add site id
temp = tabLFMC['ID'].str.split('_', 2, expand=True)
siteId = temp[0].str.cat(temp[1], sep='_')
siteIdLst = siteId.unique().tolist()
tabLFMC['siteId'] = siteId
tabSite = tabLFMC[
    ['siteId', 'Sitename', 'State/Region', 'Country', 'Latitude', 'Longitude']
].drop_duplicates()

# load LFMC from NFMD
fileLFMC2 = os.path.join(DIR_VEG, 'NFMD', 'LFMC-NFMD.csv')
tabLFMC2 = pd.read_csv(fileLFMC2, index_col=0)
tabLFMC2['date'] = pd.to_datetime(tabLFMC2['date'])
tabLFMC2 = tabLFMC2[tabLFMC2['date'] > pd.to_datetime('2015-01-01')]
tabLFMC2['lfmc'] = tabLFMC2['lfmc'] / 100
tabSite2 = tabLFMC2[['site', 'latitude', 'longitude']].drop_duplicates()

# compare site name
tabSite1 = tabSite[tabSite['Country'] == 'USA']
s1 = set(tabSite1['Sitename'].tolist())
s2 = set(tabSite2['site'].tolist())

len(s2 - s1)
len(s1 - s2)
len(s1.intersection(s2))

# plot on common ts map
lat = tabSite['Latitude'].values
lon = tabSite['Longitude'].values
extentUS = [-125, -65, 25, 50]
extentEU = [-5, 15, 40, 45]
extentGlobal = [-180, 180, -90, 90]


def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(
        figM, gsM[0, 0], lat, lon, tabSite['try_id'], extent=extentGlobal
    )
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
