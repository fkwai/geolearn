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
dfSite = pd.DataFrame(
    columns=["siteId", "siteName", "state", "fuel", "gacc", "lat", "lon"]
)
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


# modis
dirAqua = os.path.join(kPath.dirVeg, "RS", "MOD09GA-500m")
dirTerra = os.path.join(kPath.dirVeg, "RS", "MYD09GA-500m")
varM = ["sur_refl_b0{}".format(x) for x in range(1, 8)]
matM = np.full([len(tAll), len(siteIdLst), len(varM)], np.nan)
prodLst = ["Aqua", "Terra"]
dictModis = dict()
for label, dirModis in zip(prodLst, [dirAqua, dirTerra]):
    matM = np.full([len(tAll), len(siteIdLst), len(varM)], np.nan)
    for siteId in siteIdLst:
        df = pd.read_csv(os.path.join(dirModis, siteId + ".csv"))
        df.index = pd.to_datetime(df["time"]).dt.date.astype("datetime64[D]")
        df = df[varM].groupby(df.index).mean()
        temp = pd.DataFrame(index=tAll).join(df)
        matM[:, siteIdLst.index(siteId), :] = temp.values
    dictModis[label] = matM
matMA = np.concatenate([dictModis["Aqua"], dictModis["Terra"]], axis=-1)
varMA = ["mod_b{}".format(x) for x in range(1, 8)] + [
    "myd_b{}".format(x) for x in range(1, 8)
]

ns = matMA.shape[1]
b = 5
indB1 = varMA.index("mod_b{}".format(b))
indB2 = varMA.index("myd_b{}".format(b))

indS = 10
fig, ax = plt.subplots(1, 1)
ax.plot(tAll, matMA[:, indS, indB1], "*-", label="Aqua")
ax.plot(tAll, matMA[:, indS, indB2], "*-", label="Terra")
ax.legend()
fig.show()
