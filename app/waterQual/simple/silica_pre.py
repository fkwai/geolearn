from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import os
import time

# pick out sites that are have relative large number of observations
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
df0 = pd.read_csv(os.path.join(dirInv, 'codeCount.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')
df1 = pd.read_csv(os.path.join(dirInv, 'codeCount_B2000.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')
df2 = pd.read_csv(os.path.join(dirInv, 'codeCount_A2000.csv'),
                  dtype={'siteNo': str}).set_index('siteNo')

# silica num > 100 in both training and testing (named silica64)
code = '00955'
siteNoLst = df0[(df1[code] > 100) & (df2[code] > 100)].index.tolist()
wqData = waterQuality.DataModelWQ.new('Silica64', siteNoLst)
indYr1 = waterQuality.indYr(
    wqData.info, yrLst=[1979, 2000])[0]
wqData.saveSubset('Y8090', indYr1)
indYr2 = waterQuality.indYr(
    wqData.info, yrLst=[2000, 2020])[0]
wqData.saveSubset('Y0010', indYr2)


figP, axP = plt.subplots(5, 1, figsize=(8, 6))
for k in range(5):
    kk = k+5
    siteNo = siteNoLstAll[ind[kk]]
    dfC = waterQuality.readSiteY(siteNo, [code])
    t = dfC.index.values.astype(np.datetime64)
    axplot.plotTS(axP[k], t, dfC['00955'], styLst='*')
    axP[k].set_title('{} #samples = {}'.format(siteNo, dfC.count().values[0]))
figP.show()
