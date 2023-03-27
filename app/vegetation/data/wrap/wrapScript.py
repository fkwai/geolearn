import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np
import json
import os
import pandas as pd
from hydroDL import kPath

# input
nfmdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_single.json')
sd = '2016-10-15'
ed = '2021-12-15'
dataName = 'single'


# load NFMD json
tAll = pd.date_range(sd, ed, freq='D')

with open(nfmdFile, 'r') as fp:
    dictLst = json.load(fp)
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

# load NFMD data
matNFMD = np.full([len(tAll), len(dictLst)], np.nan)
for siteDict in dictLst:
    t = pd.to_datetime(siteDict['t'], format='%Y-%m-%d').values
    v = np.array(siteDict['v'])
    df = pd.DataFrame({'t': t, 'v': v}).set_index('t')
    # df = df.resample(rule='SM', label='right').mean().dropna()
    temp = pd.DataFrame({'date': tAll}).set_index('date').join(df['v'])
    k = siteIdLst.index(siteDict['siteId'])
    matNFMD[:, k] = temp['v'].values
