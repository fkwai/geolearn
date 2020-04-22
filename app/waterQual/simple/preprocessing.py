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

# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
code = '00955'

countMat = np.ndarray(len(siteNoLstAll))
for k, siteNo in enumerate(siteNoLstAll):
    print(k)
    dfC = waterQuality.readSiteY(siteNo, [code])
    countMat[k] = dfC.count().values[0]

ind = np.argsort(countMat)[::-1]
ns = np.sort(countMat)[::-1]

# get top 16 (ns>500) to a dataset
siteNoLst = [siteNoLstAll[ind[k]] for k in range(16)]
wqData = waterQuality.DataModelWQ.new('Silica16', siteNoLst)
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
