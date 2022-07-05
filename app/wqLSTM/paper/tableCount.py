
from socket import AddressFamily
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib

dataName = 'N200'

DF = dbBasin.DataFrameBasin(dataName)
codeLst = usgs.varC

trainLst = ['rmRT20', 'rmL20', 'rmYr5']
testLst = ['pkRT20', 'pkL20', 'pkYr5']

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)
matS = count >= 200

cols = ['code', 'name', 'abv',
        'unit', 'nObs', 'nSite200', 'nsiteYr5']
tab = pd.DataFrame(columns=cols)
tab['code'] = codeLst
tab['name'] = usgs.codePdf.loc[codeLst]['srsName'].values
tab['abv'] = usgs.codePdf.loc[codeLst]['shortName'].values
tab['unit'] = usgs.codePdf.loc[codeLst]['unit'].values
tab['nObs'] = np.mean(count, axis=0)
tab['nSite200'] = np.sum(matS, axis=0)

for trainSet, testSet in zip(trainLst, testLst):
    matB1 = DF.extractSubset(matB, trainSet)
    matB2 = DF.extractSubset(matB, testSet)
    count1 = np.nansum(matB1, axis=0)
    count2 = np.nansum(matB2, axis=0)
    if trainSet == 'rmYr5':
        matP = (count1 >= 80) & (count2 >= 20)
    else:
        matP = (count1 >= 120) & (count2 >= 30)
    tab['nsite{}'.format(trainSet[2:])] = np.sum(matP, axis=0)


outFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
tab.to_csv(os.path.join(outFolder, 'tabCount.csv'))
