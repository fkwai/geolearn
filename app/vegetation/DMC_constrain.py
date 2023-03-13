import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
import numpy as np
from hydroDL import kPath
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

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
siteLFMC = (
    tabData[['siteId', 'Latitude', 'Longitude']]
    .drop_duplicates()
    .set_index('siteId')
    .sort_index()
)

# add spec id
fileSpec = os.path.join(DIR_VEG, 'specMatch', 'summary-LFMC')
dfFix = pd.read_csv(fileSpec, index_col=0)
tab = tabData.merge(dfFix, left_on='Species collected', right_index=True, how='left')

# load DMC
fileDMC = os.path.join(DIR_VEG, 'TRY', 'DMC.csv')
tabDMC = pd.read_csv(fileDMC)
tabDMC['siteId'] = tabDMC.groupby(['Latitude', 'Longitude']).grouper.group_info[0]
siteDMC = (
    tabDMC[['siteId', 'Latitude', 'Longitude']]
    .drop_duplicates()
    .set_index('siteId')
    .sort_index()
)

# plot on one species
tryId = 4447
outFolder = os.path.join(DIR_VEG, 'map-DMC-LFMC')
tryIdLst = tabDMC['AccSpeciesID'].unique().tolist()
tryIdLst.remove(0)

for tryId in tryIdLst:
    tabD = tabDMC[tabDMC['AccSpeciesID'] == tryId]
    tabL = tab[tab['try_id'] == tryId]
    if len(tabD) > 0 and len(tabL) > 0:
        fileName = 'L{}-D{}-T{}'.format(len(tabL), len(tabD), tryId)

        tabDM = tabD.groupby(['siteId'])['StdValue'].mean()
        maxD = (1 - tabDM) / tabDM
        maxL = tabL.groupby(['siteId'])['LFMC value'].max()
        maxL2 = tabL.groupby(['siteId'])['LFMC value'].quantile(0.9)

        lat1 = siteDMC.loc[maxD.index]['Latitude'].values
        lon1 = siteDMC.loc[maxD.index]['Longitude'].values
        v1 = maxD.values
        lat2 = siteLFMC.loc[maxL.index]['Latitude'].values
        lon2 = siteLFMC.loc[maxL.index]['Longitude'].values
        v2 = maxL.values / 100
        v3 = maxL2.values / 100
        temp = np.concatenate([v1, v2])
        tempLat = np.concatenate([lat1, lat2])
        tempLon = np.concatenate([lon1, lon2])
        [x1, x2, y1, y2] = [
            np.nanmin(tempLon) - 1,
            np.nanmax(tempLon) + 1,
            np.nanmin(tempLat) - 1,
            np.nanmax(tempLat) + 1,
        ]
        vRange = [np.percentile(temp, 10), np.percentile(temp, 90)]

        if x1 < -30 and x2 > -10:
            extent1 = [x1, np.nanmax(tempLon[tempLon < -30]) + 1, y1, y2]
            extent2 = [np.nanmin(tempLon[tempLon > 10]) - 1, x2, y1, y2]
            fig = plt.figure(figsize=(16, 8))
            gs = gridspec.GridSpec(2, 3)
            ax1 = mapplot.mapPoint(
                fig, gs[0, 0], lat1, lon1, v1, vRange=vRange, extent=extent1
            )
            ax2 = mapplot.mapPoint(
                fig, gs[0, 1], lat2, lon2, v2, vRange=vRange, extent=extent1
            )
            ax3 = mapplot.mapPoint(
                fig, gs[0, 2], lat2, lon2, v3, vRange=vRange, extent=extent1
            )
            ax4 = mapplot.mapPoint(
                fig, gs[1, 0], lat1, lon1, v1, vRange=vRange, extent=extent2
            )
            ax5 = mapplot.mapPoint(
                fig, gs[1, 1], lat2, lon2, v2, vRange=vRange, extent=extent2
            )
            ax6 = mapplot.mapPoint(
                fig, gs[1, 2], lat2, lon2, v3, vRange=vRange, extent=extent2
            )
            gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False
            gl = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False
        else:
            extent = [x1, x2, y1, y2]
            fig = plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, 3)
            ax1 = mapplot.mapPoint(
                fig, gs[0, 0], lat1, lon1, v1, vRange=vRange, extent=extent
            )
            ax2 = mapplot.mapPoint(
                fig, gs[0, 1], lat2, lon2, v2, vRange=vRange, extent=extent
            )
            ax3 = mapplot.mapPoint(
                fig, gs[0, 2], lat2, lon2, v3, vRange=vRange, extent=extent
            )
            gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False
        ax1.set_title('upper limit of LFMC of DMC sites')
        ax2.set_title('max of LFMC')
        ax3.set_title('p90 of LFMC')
        # fig.show()
        fig.savefig(os.path.join(outFolder, fileName))
