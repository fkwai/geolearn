from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins

import pandas as pd
import numpy as np
import os
import time

# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist()
             if siteNo in siteNoLstAll]

# shapefiles
usgsDir = os.path.join(kPath.dirData, 'USGS')
outShapeFile = os.path.join(usgsDir, 'basins', 'HBN.shp')
# gageII.extractBasins(siteNoHBN, outShapeFile)

# wrap up data
if not waterQuality.exist('HBN'):
    wqData = waterQuality.DataModelWQ.new('HBN', siteNoHBN)
if not waterQuality.exist('HBN-30d'):
    wqData = waterQuality.DataModelWQ.new('HBN-30d', siteNoHBN, rho=30)
if not waterQuality.exist('HBN-5s'):
    wqData = waterQuality.DataModelWQ.new('HBN-5s', siteNoHBN[:5])
if not waterQuality.exist('HBN-5s-30d'):
    wqData = waterQuality.DataModelWQ.new('HBN-5s-30d', siteNoHBN[:5], rho=30)

# wrap up data
if not waterQuality.exist('HBN'):
    wqData = waterQuality.DataModelWQ.new('HBN', siteNoHBN)
else:
    wqData = waterQuality.DataModelWQ('HBN')
if 'first80-rm2' not in wqData.subset.keys():
    ind = wqData.subset['first80']
    indRm = wqData.indByComb(['00010', '00095'])
    indTrain = np.setdiff1d(ind, indRm)
    wqData.saveSubset('first80-rm2', indTrain)

if 'first50' not in wqData.subset.keys():
    ind1 = wqData.indByRatio(0.5)
    ind2 = wqData.indByRatio(0.5, first=False)
    wqData.saveSubset(['first50', 'last50'], [ind1, ind2])


# divide subsets based on years
wqData = waterQuality.DataModelWQ('HBN')
info = wqData.info
info['yr'] = pd.DatetimeIndex(info['date']).year
yrLst = [1979,1990,2000,2010,2020]
subsetLst = ['80s', '90s', '00s', '10s']
indLst = list()
indAll = info.index.values
for k in range(len(yrLst)-1):
    sy = yrLst[k]
    ey = yrLst[k+1]
    ind = info.index[(info['yr'] >= sy) & (info['yr'] < ey)].values
    indLst.append(ind)
for ind, subset in zip(indLst, subsetLst):
    wqData.saveSubset(subset, ind)
    indRm = np.setdiff1d(indAll, ind)
    wqData.saveSubset(subset+'-rm', indRm)
for subset in subsetLst:
    len(wqData.subset[subset])
wqData.info.iloc[wqData.subset['80s-rm']]['yr'].unique()