import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import utils
import numpy as np
import json
import os
import pandas as pd
from hydroDL import kPath
from osgeo import ogr, gdal

# input
nfmdFile = os.path.join(kPath.dirVeg, "NFMD", "NFMD_single.json")
sd = "2016-10-15"
ed = "2021-12-15"
dataName = "single"

# load NFMD json
tAll = pd.date_range(sd, ed, freq="D")
with open(nfmdFile, "r") as fp:
    dictLst = json.load(fp)
dfSite = pd.DataFrame(columns=["siteId", "siteName", "state", "fuel", "gacc", "lat", "lon"])
for k, siteDict in enumerate(dictLst):
    dfSite.loc[k] = [
        siteDict["siteId"],
        siteDict["siteName"],
        siteDict["state"],
        siteDict["fuel"],
        siteDict["gacc"],
        siteDict["crd"][0],
        siteDict["crd"][1],
    ]
dfSite = dfSite.set_index("siteId").sort_index()
siteIdLst = dfSite.index.to_list()

# check swath of RS
scale = 500
dirSentinel = os.path.join(kPath.dirVeg, "RS", "sentinel1-{}m".format(scale))


siteId = siteIdLst[0]
matV = np.full([len(siteIdLst), 2], np.nan)
matC = np.full([len(siteIdLst), 2], np.nan)
for k, siteId in enumerate(siteIdLst):
    df = pd.read_csv(os.path.join(dirSentinel, siteId + ".csv"))
    angle = np.round(df['angle'].values)
    v, count = np.unique(angle, return_counts=True)
    if len(v) == 2:
        matV[k, :] = v
        matC[k, :] = count
    elif len(v) > 2:
        # find between 30 and 40
        ind1 = np.where((v >= 30) & (v <= 40))[0]
        ind2 = np.where((v >= 40) & (v <= 50))[0]
        vNew = [v[ind1[0]], v[ind2[0]]]
        countNew = [np.sum(count[ind1]), np.sum(count[ind2[0]])]
        matV[k, :] = vNew
        matC[k, :] = countNew
        print(v, count)
    elif len(v) == 1:
        if v[0] < 40:
            matV[k, :] = [v[0], np.nan]
            matC[k, :] = [count[0], 0]
        else:
            matV[k, :] = [np.nan, v[0]]
            matC[k, :] = [0, count[0]]

# stacked bar plot for two count and two values

fig, ax = plt.subplots(1, 1)
p1 = ax.bar(np.arange(matC.shape[0]), matC[:, 0], label="angle ~35")
p2 = ax.bar(np.arange(matC.shape[0]), matC[:, 1], bottom=matC[:, 0], label="angle ~45")

fig.legend()
fig.show()


# sentinel angle
from matplotlib import cm

y = np.array([1, 4, 3, 2, 7, 11])
colors = cm.hsv(y / float(max(y)))
plot = plt.scatter(y, y, c=y, cmap='hsv')
plt.clf()
plt.colorbar(plot)
plt.bar(range(len(y)), y, color=colors)
plt.show()
