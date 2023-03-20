import os
import pandas as pd
from hydroDL import kPath
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np
import json

outFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD.csv')
tabAll = pd.read_csv(outFile)
tabAll['Date'] = pd.to_datetime(tabAll['Date'], format='%Y-%m-%d')
tabAll = tabAll[tabAll['Date'] > '2019-07-01']
crdFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMDsite.csv')
tabCrd = pd.read_csv(crdFile, index_col=0)

# find sites that are of single fuel
cntFuel = tabAll.groupby(['siteId'])['Fuel'].nunique().sort_values()
siteSingle = cntFuel[cntFuel == 1].index.tolist()
siteMix = cntFuel[cntFuel > 1].index.tolist()

# outFile
singleFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_single_A19.json')
mixFile = os.path.join(kPath.dirVeg, 'NFMD', 'NFMD_mix_A19.json')

# single sites to dict
dictSingle = list()
for k, site in enumerate(siteSingle):
    print(k, site)
    tab = tabAll[tabAll['siteId'] == site]
    tab = tab.sort_values(by=['Date'])
    crd = tabCrd.loc[site]
    if not np.isnan(crd['lat']):
        siteDict = dict(
            siteId=site,
            siteName=crd['Site'],
            state=crd['State'],
            fuel=tab['Fuel'].iloc[0],
            gacc=crd['GACC'],
            crd=[crd['lat'], crd['lon']],
            t=[d.strftime('%Y-%m-%d') for d in tab['Date'].tolist()],
            v=tab['Percent'].tolist(),
        )
        dictSingle.append(siteDict)
# save to json
with open(singleFile, 'w') as fp:
    json.dump(dictSingle, fp, indent=4)

# mix sites to dict
dictMix = list()
for k, site in enumerate(siteMix):
    print(k, site)
    crd = tabCrd.loc[site]
    tab = tabAll[tabAll['siteId'] == site]
    tab = tab.sort_values(by=['Date'])
    fuelLst = tab['Fuel'].unique().tolist()
    tLst = list()
    vLst = list()
    for fuel in fuelLst:
        tabTemp = tab[tab['Fuel'] == fuel]
        tLst.append(tabTemp['Date'].tolist())
        vLst.append(tabTemp['Percent'].tolist())

    tOut, indLst = utils.intersectMulti(tLst)
    if len(tOut) > 10:
        mat = np.zeros([len(fuelLst), len(tOut)])
        for k, ind in enumerate(indLst):
            mat[k, :] = np.array(vLst[k])[ind]
        corrMat = np.corrcoef(mat)
    else:
        corrMat = np.zeros([len(fuelLst), len(fuelLst)])
    if (corrMat > 0.5).all() and (not np.isnan(crd['lat'])):
        vOut = np.mean(mat, axis=0)
        siteDict = dict(
            siteId=site,
            siteName=crd['Site'],
            state=crd['State'],
            gacc=crd['GACC'],
            crd=[crd['lat'], crd['lon']],
            t=[d.strftime('%Y-%m-%d') for d in tOut],
            v=vOut.tolist(),
            fuel=fuelLst,
            minCorr=np.min(corrMat),
        )
        dictMix.append(siteDict)
with open(mixFile, 'w') as fp:
    json.dump(dictMix, fp, indent=4)

# fig, ax = plt.subplots(1, 1)
# for t, v in zip(tLst, vLst):
#     ax.plot(t, v)
# fig.show()


# load from json
# with open(outFile, 'r') as fp:
#     dictLst = json.load(fp)
#     for siteDict in dictLst:
#         siteDict['t'] = pd.to_datetime(siteDict['t'], format='%Y-%m-%d').values
#         siteDict['v'] = np.array(siteDict['v'])
