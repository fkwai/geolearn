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
tAll = pd.date_range(sd, ed, freq='SM')

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

for siteDict in dictLst:
    t = pd.to_datetime(siteDict['t'], format='%Y-%m-%d').values
    v = np.array(siteDict['v'])
    df = pd.DataFrame({'t': t, 'v': v}).set_index('t')
    df = df.resample(rule='SM', label='right').mean().dropna()
    temp = pd.DataFrame({'date': tAll}).set_index('date').join(df['v'])
    k = siteIdLst.index(siteDict['siteId'])
    matV[:, k] = temp['v'].values

1 - np.sum(np.isnan(matV)) / (matV.shape[0] * matV.shape[1])

# RS


def interpSM(inputDF, var):
    df = inputDF.copy()
    df = df[var]
    df = df.dropna()
    df = df.resample('SM').mean()
    df = df.interpolate(limit=8)
    return df


# landsat
dictCol = {
    'SR_B2': 'blue',
    'SR_B3': 'green',
    'SR_B4': 'red',
    'SR_B5': 'nir',
    'SR_B6': 'swir',
}
matL = np.full([len(tAll), len(siteIdLst), len(dictCol)], np.nan)
for siteId in siteIdLst:
    dirLandsat = os.path.join(kPath.dirVeg, 'RS', 'landsat8-500m')
    df = pd.read_csv(os.path.join(dirLandsat, siteId + '.csv'))
    df = df.rename(columns=dictCol)
    # drop cloudy days
    cloud_shadow = [328, 392, 840, 904, 1350]
    cloud = [352, 368, 416, 432, 480, 864, 880, 928, 944, 992]
    high_confidence_cloud = [480, 992]
    all_masked_values = cloud_shadow + cloud + high_confidence_cloud
    # interpolate
    df = df[~df['QA_PIXEL'].isin(all_masked_values)]
    df.index = pd.to_datetime(df['time'])
    dfi = interpSM(df, var=dictCol.values())
    temp = pd.DataFrame(index=tAll).join(dfi)
    # temp=pd.merge(pd.DataFrame(index=tAll),dfi,left_index=True,right_index=True)
    matL[:, siteIdLst.index(siteId), :] = temp.values
1 - np.sum(np.isnan(matL)) / (len(matL.flatten()))

# load sentinel
dirSentinel = os.path.join(kPath.dirVeg, 'RS', 'sentinel1-500m')
varLst = ['VV', 'VH']
matS = np.full([len(tAll), len(siteIdLst), len(varLst)], np.nan)
for siteId in siteIdLst:
    df = pd.read_csv(os.path.join(dirSentinel, siteId + '.csv'))
    df.index = pd.to_datetime(df['time'])
    dfi = interpSM(df, var=varLst)
    temp = pd.DataFrame(index=tAll).join(dfi)
    matS[:, siteIdLst.index(siteId), :] = temp.values
1 - np.sum(np.isnan(matS)) / (len(matS.flatten()))

# imageshow
imgL = np.sum(~np.isnan(matL), axis=-1)
imgS = np.sum(~np.isnan(matS), axis=-1)
indT = np.where((tAll.month.isin([3, 6, 9, 12])) & (tAll.day == 15))[0]
fig, axes = plt.subplots(1, 1, figsize=(12, 4))
# axes[0].set_yticks(indT)
# axes[0].set_yticklabels(tAll[indT].strftime('%Y-%m-%d'))
# axes[0].imshow(imgL, aspect='auto')
# axes[1].set_yticks(indT)
# axes[1].set_yticklabels(tAll[indT].strftime('%Y-%m-%d'))
# axes[1].imshow(imgS, aspect='auto')
axes.set_yticks(indT)
axes.set_yticklabels(tAll[indT].strftime('%Y-%m-%d'))
axes.imshow(imgS, aspect='auto')
fig.show()

# constrain period
sdN = '2016-08-31'
edN = '2021-12-15'
tN = pd.date_range(sdN, edN, freq='SM')
_, indT1, indT2 = np.intersect1d(tAll, tN, return_indices=True)
matVN = matV[indT1, :]
matLN = matL[indT1, :, :]
matSN = matS[indT1, :, :]

varY = 'lfmc'
y = matVN / 100
varX = ['blue', 'green', 'red', 'nir', 'swir', 'VV', 'VH']
x = np.concatenate([matLN, matSN], axis=-1)
info = dfSite

# save
outFile = os.path.join(kPath.dirVeg, 'model', 'data', 'trainData.npz')
np.savez(outFile, varY=varY, y=y, varX=varX, x=x, t=tN)
dfSite.to_csv(os.path.join(kPath.dirVeg, 'model', 'data', 'site.csv'))
1 - np.sum(np.isnan(matSN)) / (len(matSN.flatten()))
1 - np.sum(np.isnan(matS)) / (len(matS.flatten()))
