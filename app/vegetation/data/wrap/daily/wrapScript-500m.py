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
dataName = "singleDaily-MG"
# gridName = 'modisgrid'
gridName = 'nadgrid'

dirLandsat = os.path.join(kPath.dirVeg, "RS", "landsat8-{}".format(gridName))
dirSentinel = os.path.join(kPath.dirVeg, "RS", "sentinel1-{}".format(gridName))
dirModis = os.path.join(kPath.dirVeg, "RS", "MCD43A4-{}".format(gridName))


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

# load NFMD data
matNFMD = np.full([len(tAll), len(dictLst)], np.nan)
for siteDict in dictLst:
    t = pd.to_datetime(siteDict["t"], format="%Y-%m-%d").values
    v = np.array(siteDict["v"])
    df = pd.DataFrame({"t": t, "v": v}).set_index("t")
    # df = df.resample(rule='SM', label='right').mean().dropna()
    temp = pd.DataFrame({"date": tAll}).set_index("date").join(df["v"])
    k = siteIdLst.index(siteDict["siteId"])
    matNFMD[:, k] = temp["v"].values

# landsat
varL = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"]
matL = np.full([len(tAll), len(siteIdLst), len(varL)], np.nan)
for siteId in siteIdLst:
    df = pd.read_csv(os.path.join(dirLandsat, siteId + ".csv"))
    df.index = pd.to_datetime(df["time"]).dt.date.astype("datetime64[D]")
    df = df[varL].groupby(df.index).mean()
    temp = pd.DataFrame(index=tAll).join(df)
    matL[:, siteIdLst.index(siteId), :] = temp.values

# sentinel
varS = ["VV", "VH"]
matS = np.full([len(tAll), len(siteIdLst), len(varS)], np.nan)
for siteId in siteIdLst:
    df = pd.read_csv(os.path.join(dirSentinel, siteId + ".csv"))
    df.index = pd.to_datetime(df["time"]).dt.date.astype("datetime64[D]")
    df = df[varS].groupby(df.index).mean()
    temp = pd.DataFrame(index=tAll).join(df)
    matS[:, siteIdLst.index(siteId), :] = temp.values

# modis Adjusted
varM = ["Nadir_Reflectance_Band{}".format(x) for x in range(1, 8)]
matM = np.full([len(tAll), len(siteIdLst), len(varM)], np.nan)
for siteId in siteIdLst:
    df = pd.read_csv(os.path.join(dirModis, siteId + ".csv"))
    df.index = pd.to_datetime(df["time"]).dt.date.astype("datetime64[D]")
    df = df[varM].groupby(df.index).mean()
    temp = pd.DataFrame(index=tAll).join(df)
    matM[:, siteIdLst.index(siteId), :] = temp.values
varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]

# read forcing
varF = ["pr", "sph", "srad", "tmmn", "tmmx", "pet", "etr"]
matF = np.full([len(tAll), len(dfSite), len(varF)], np.nan)
for k, var in enumerate(varF):
    dfM = pd.read_csv(os.path.join(kPath.dirVeg, "forcings", "{}.csv".format(var)), index_col=0)
    dfM.index = pd.to_datetime(dfM.index)
    tM = dfM.index.values
    _, indT1, indT2 = np.intersect1d(tM, tAll, return_indices=True)

    v = dfM[dfSite.index].values[indT1, :]
    matF[:, :, k] = v


# pre-processed constant
def get_value(filename, mx, my):
    ds = gdal.Open(filename)
    gt = ds.GetGeoTransform()
    data = ds.GetRasterBand(1).ReadAsArray()
    px = ((mx - gt[0]) / gt[1]).astype(int)  # x pixel
    py = ((my - gt[3]) / gt[5]).astype(int)  # y pixel
    return data[py, px]


dictConst = {
    "slope": "usa_slope_project.tif",
    "dem": "usa_dem.tif",
    "canopyHeight": "canopy_height.tif",
    "sand": "Unified_NA_Soil_Map_Subsoil_Sand_Fraction.tif",
    "clay": "Unified_NA_Soil_Map_Subsoil_Clay_Fraction.tif",
    "silt": "Unified_NA_Soil_Map_Subsoil_Silt_Fraction.tif",
}
siteFile = os.path.join(kPath.dirVeg, "model", "data", "site.csv")
dfSite = pd.read_csv(siteFile)

lat = dfSite["lat"].values
lon = dfSite["lon"].values
matConst = np.full([len(dfSite), len(dictConst)], np.nan)
for k, (key, value) in enumerate(dictConst.items()):
    temp = get_value(os.path.join(kPath.dirVeg, "const", value), lon, lat)
    matConst[:, k] = temp
varConst = list(dictConst.keys())

# land cover
fileName = os.path.join(kPath.dirRaw, "NLCD", "2016", "nlcd_2016.tif")
matLC = np.zeros([len(lat), 9])
ds = gdal.Open(fileName)
gt = ds.GetGeoTransform()
data = ds.GetRasterBand(1).ReadAsArray()
pxA = ((lon - gt[0]) / gt[1]).astype(int)  # x pixel
pyA = ((lat - gt[3]) / gt[5]).astype(int)  # y pixel
for kk, (px, py) in enumerate(zip(pxA, pyA)):
    n = 5
    temp = data[px - n : px + n + 1, py - n : py + n + 1]
    temp = np.floor(temp / 10)
    v = np.zeros(9)
    for k in range(9):
        v[k] = np.sum(temp == k + 1)
    v = v / (n * 2 + 1) ** 2
    matLC[kk, :] = v
varLC = ["lc" + str(x) for x in range(1, 10)]

# additional RS fields
dictCol = {
    "SR_B2": "blue",
    "SR_B3": "green",
    "SR_B4": "red",
    "SR_B5": "nir",
    "SR_B6": "swir",
}
red = matL[:, :, varL.index("SR_B4")]
nir = matL[:, :, varL.index("SR_B5")]
swir = matL[:, :, varL.index("SR_B6")]
ndvi = (nir - red) / (nir + red)
ndwi = (nir - swir) / (nir + swir)
nirv = nir * ndvi
vh = matS[:, :, varS.index("VH")]
vv = matS[:, :, varS.index("VV")]
vh_vv = vh - vv
matLA = np.dstack([matL, ndvi, ndwi, nirv])
varLA = varL + ["ndvi", "ndwi", "nirv"]
matSA = np.dstack([matS, vh_vv])
varSA = varS + ["vh_vv"]


# combine
varX = varSA + varLA + varM + varF
matX = np.concatenate([matSA, matLA, matM, matF], axis=-1)
varY = ["LFMC"]
matY = matNFMD[:, :, None]
varXC = varConst + varLC
matXC = np.concatenate([matConst, matLC], axis=-1)

outFolder = os.path.join(kPath.dirVeg, "model", "data", dataName)
if not os.path.exists(outFolder):
    os.mkdir(outFolder)
outFile = os.path.join(outFolder, "data.npz")
np.savez(outFile, t=tAll, varY=varY, y=matY, varX=varX, x=matX, varXC=varXC, xc=matXC)
dfSite.to_csv(os.path.join(outFolder, "site.csv"), index=False)
