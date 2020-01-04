import importlib
from hydroDL.data import gridMET
from hydroDL import kPath
import numpy as np
import pandas as pd
import os
import time
import argparse
"""
convert raw data to tab format of each sites
"""

workDir = kPath.dirWQ
dataFolder = os.path.join(kPath.dirData, 'gridMET')
maskFolder = os.path.join(kPath.dirData, 'USGS-mask')
rawFolder = os.path.join(kPath.dirData, 'USGS-gridMET', 'raw')
saveFolder = os.path.join(kPath.dirData, 'USGS-gridMET')

# create mask for all USGS basins: gridMetMask.py
# extract gridMet data for all USGS basins: gridMetExtract.py

# setup information
varLst = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
yrLst = range(1979, 2020)
fileSiteNo = os.path.join(workDir, 'modelUsgs2', 'siteNoSel')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
t, _, _ = gridMET.readNcInfo(os.path.join(dataFolder, 'pr_1979.nc'))
sd = t[0]
t, _, _ = gridMET.readNcInfo(os.path.join(dataFolder, 'pr_2019.nc'))
ed = t[-1]
tInd = pd.date_range(start=sd, end=ed)

# read all data
dataDict = dict()
for var in varLst:
    dataLst = list()
    for yr in yrLst:
        file = os.path.join(rawFolder, '{}_{}.npy'.format(var, yr))
        data = np.load(file)
        dataLst.append(data)
    dataDict[var] = np.concatenate(dataLst)
    print('reading '+var)

# write to files
for k, siteNo in enumerate(siteNoLst):
    siteNo = siteNoLst[k]
    temp = dict()
    temp['date'] = tInd
    for var in varLst:
        temp[var] = dataDict[var][:, k]
    pdf = pd.DataFrame.from_dict(temp)
    fileName = os.path.join(saveFolder, siteNo)
    pdf.to_csv(fileName, index=False)
    print('writing '+str(k)+fileName)
