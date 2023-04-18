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
ed = '2023-03-01'
tAll = pd.date_range(sd, ed, freq='SM')

# load NFMD json
with open(singleFile, 'r') as fp:
    dictLst = json.load(fp)

# init
matV = np.full([len(tAll), len(dictLst)], np.nan)
dfSite = pd.DataFrame(
    columns=['siteId', 'siteName', 'state', 'fuel', 'gacc', 'lat', 'lon']
)

for k, siteDict in enumerate(dictLst):
    t = pd.to_datetime(siteDict['t'], format='%Y-%m-%d').values
    v = np.array(siteDict['v'])
    df = pd.DataFrame({'t': t, 'v': v}).set_index('t')
    df = df.resample(rule='SM', label='right').mean().dropna()
    temp = pd.DataFrame({'date': tAll}).set_index('date').join(df['v'])
    matV[:, k] = temp['v'].values
    dfSite.loc[k] = [
        siteDict['siteId'],
        siteDict['siteName'],
        siteDict['state'],
        siteDict['fuel'],
        siteDict['gacc'],
        siteDict['crd'][0],
        siteDict['crd'][1],
    ]

1 - np.sum(np.isnan(matV)) / (matV.shape[0] * matV.shape[1])

# RS


def interpSM(inputDF, var):
    df = inputDF.copy()
    df = df.dropna()
    df = df.resample('SM').mean()
    df = df.interpolate(limit=8)
    df = df[var]
    return df


# landsat
siteIdLst = dfSite['siteId'].to_list()
siteId = siteIdLst[0]
dirLandsat = os.path.join(kPath.dirVeg, 'RS', 'landsat8-500m')
dictCol = {'SR_B2': 'blue', 'SR_B3': 'green', 'SR_B4': 'red', 'SR_B5': 'nir', 'SR_B6': 'swir'}
df = pd.read_csv(os.path.join(dirLandsat, siteId + '.csv'))
df = df.rename(columns=dictCol)
# drop cloudy days
cloud_shadow = [328, 392, 840, 904, 1350]
cloud = [352, 368, 416, 432, 480, 864, 880, 928, 944, 992]
high_confidence_cloud = [480, 992]
all_masked_values = cloud_shadow + cloud + high_confidence_cloud
# interpolate - SM
df = df[~df['QA_PIXEL'].isin(all_masked_values)]
df.index = pd.to_datetime(df['time']).dt.date.astype('datetime64[D]')
dfi = interpSM(df, var=dictCol.values())

# interpolate daily - SM
tDaily = pd.date_range(sd, ed, freq='D')
dfD = pd.DataFrame({'date': tDaily}).set_index('date').join(df)
dfD = dfD.interpolate(limit=8)
dfD=dfD.resample('SM').mean()


fig, ax = plt.subplots(1, 1)
ax.plot(df.index, df['nir'], 'r*-', label='raw')
ax.plot(dfi.index, dfi['nir'], 'b*-', label='SM')
ax.plot(dfD.index, dfD['nir'], 'g*-', label='SM-New')
fig.show()

# load sentinel
dirSentinel = os.path.join(kPath.dirVeg, 'RS', 'sentinel1-500m')
df = pd.read_csv(os.path.join(dirSentinel, siteId + '.csv'))

feature_sub = interpolate(
    df_sub, var=optical_inputs, resolution=resolution, max_gap=max_gap
)
