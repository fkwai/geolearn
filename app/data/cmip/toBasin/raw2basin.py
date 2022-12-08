import importlib
from hydroDL.data.cmip import io
from hydroDL import kPath
import numpy as np
import pandas as pd
import os
import time
import argparse
import json

mLst = ['MPI-ESM1-2-XR', 'EC-Earth3P-HR', 'EC-Earth3P', 'CNRM-CM6-1-HR']
modelName = 'MPI-ESM1-2-XR'

dataFolder = os.path.join(kPath.dirRaw, 'CMIP', modelName)
outFolder = os.path.join(kPath.dirData, 'USGS','CMIP', modelName)
maskFolder = os.path.join(kPath.dirData, 'USGS', 'mask', 'CMIP', modelName)
rawFolder = os.path.join(outFolder, 'raw')
saveFolder = os.path.join(outFolder, 'output')

if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'basins', 'siteNoLst.json')
with open(fileSiteNo) as fp:
    dictSite = json.load(fp)
siteNoLst = dictSite['CONUS']

var = 'pr'
lab = 'hist-1950'
yLst = list(range(1980, 2015, 5))
sLst = list(range(0, len(siteNoLst), 2000))
varLst = ['pr', 'tasmax', 'tasmin']

for s1, s2 in zip(sLst[:-1], sLst[1:]):
    errLst = list()
    dataDict = dict()
    for var in varLst:
        tempLst = list()
        for y1, y2 in zip(yLst[:-1], yLst[1:]):
            tempName = '{}_{}_{}_{}_{}.csv'.format(var, y1, y2, s1, s2)
            fileName = os.path.join(rawFolder, tempName)
            temp = pd.read_csv(fileName, index_col=0)
            tempLst.append(temp)
        pdf = pd.concat(tempLst)
        dataDict[var] = pdf
    siteNoTemp = pdf.columns.tolist()
    t = pd.to_datetime(pdf.index).values.astype('datetime64[D]')
    for siteNo in siteNoTemp:
        temp = dict()
        temp['date'] = t
        for var in varLst:
            temp[var] = dataDict[var][siteNo]
        pdfSite = pd.DataFrame.from_dict(temp).set_index('date')
        pdfSite.to_csv(os.path.join(saveFolder, siteNo))
        print('writing '+siteNo)
        if pd.isna(pdfSite).all().all():
            errLst.append(siteNo)
