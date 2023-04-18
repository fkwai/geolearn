import os
import pandas as pd
from hydroDL import kPath
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np
import json

singleFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_single.json')
mixFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_mix.json')

sd = '2014-12-31'
ed = '2023-02-15'
tAll = pd.date_range(sd, ed)

# load NFMD json
with open(singleFile, 'r') as fp:
    dictLst = json.load(fp)

# init site df
matV = np.full([len(tAll), len(dictLst)], np.nan)
dfSite = pd.DataFrame(
    columns=['siteId', 'siteName', 'state', 'fuel', 'gacc', 'lat', 'lon']
)
for k, siteDict in enumerate(dictLst):
    dfSite.loc[k] = [
        siteDict['siteId'],
        siteDict['siteName'],
        siteDict['state'],
        siteDict['fuel'],
        siteDict['gacc'],
        siteDict['crd'][0],
        siteDict['crd'][1],
    ]
dfSite = dfSite.set_index('siteId').sort_index()
siteIdLst = dfSite.index.to_list()


# landsat
varLst = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6']
dirLandsat = os.path.join(kPath.dirVeg, 'RS', 'landsat8-500m')
df = pd.read_csv(os.path.join(dirLandsat, siteIdLst[20] + '.csv'))
# tL = pd.to_datetime(df['time']).dt.date.astype('datetime64[D]').drop_duplicates()
# (tL[1:] - tL[:-1]).unique()
matL = np.full([len(tAll), len(siteIdLst), len(varLst)], np.nan)
for siteId in siteIdLst:
    df = pd.read_csv(os.path.join(dirLandsat, siteId + '.csv'))    
    # drop cloudy days
    cloud_shadow = [328, 392, 840, 904, 1350]
    cloud = [352, 368, 416, 432, 480, 864, 880, 928, 944, 992]
    high_confidence_cloud = [480, 992]
    all_masked_values = cloud_shadow + cloud + high_confidence_cloud
    # interpolate
    df = df[~df['QA_PIXEL'].isin(all_masked_values)]    
    df.index = pd.to_datetime(df['time']).dt.date.astype('datetime64[D]')
    df=df[varLst].groupby(df.index).mean()
    temp = pd.DataFrame(index=tAll).join(df[varLst])
    # temp=pd.merge(pd.DataFrame(index=tAll),dfi,left_index=True,right_index=True)
    matL[:, siteIdLst.index(siteId), :] = temp.values
1 - np.sum(np.isnan(matL)) / (len(matL.flatten()))

# load sentinel
dirSentinel = os.path.join(kPath.dirVeg, 'RS', 'sentinel1-500m')
varLst = ['VV', 'VH']
matS = np.full([len(tAll), len(siteIdLst), len(varLst)], np.nan)
for siteId in siteIdLst:
    df = pd.read_csv(os.path.join(dirSentinel, siteId + '.csv'))
    df.index = pd.to_datetime(df['time']).dt.date.astype('datetime64[D]')
    df=df[varLst].groupby(df.index).mean()
    temp = pd.DataFrame(index=tAll).join(df)
    matS[:, siteIdLst.index(siteId), :] = temp.values
1 - np.sum(np.isnan(matS)) / (len(matS.flatten()))

# imageshow
imgL = np.sum(~np.isnan(matL), axis=-1)
imgS = np.sum(~np.isnan(matS), axis=-1)
indT = np.where((tAll.day == 15))[0]
fig, axes = plt.subplots(2, 1, figsize=(12, 4))
axes[0].set_yticks(indT)
axes[0].set_yticklabels(tAll[indT].strftime('%Y-%m-%d'))
im1=axes[0].imshow(imgL, aspect='auto')
axes[1].set_yticks(indT)
axes[1].set_yticklabels(tAll[indT].strftime('%Y-%m-%d'))
im2=axes[1].imshow(imgS, aspect='auto')
fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
fig.show()
