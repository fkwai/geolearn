from hydroDL import kPath
from hydroDL.app import waterQuality
import pandas as pd
import numpy as np
import time
import os
import hydroDL.data.usgs.read as read

"""
extract sites that are in raw folder but not in csv folder
"""
dirUSGS = kPath.dirUsgs
siteNoFile = os.path.join(kPath.dirUsgs, 'basins', 'siteCONUS.csv')
dfSite = pd.read_csv(siteNoFile, dtype={'siteNo': str})
siteNoLst = dfSite['siteNo'].tolist()

dirQ = os.path.join(kPath.dirUsgs, 'streamflow', 'csv')
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    fileCsv = os.path.join(dirQ, siteNo)
    if not os.path.exists(fileCsv):
        dfQ = read.readStreamflow(siteNo, csv=False)
        if dfQ is not None:
            dfQ.to_csv(os.path.join(dirQ, siteNo))
        print('\t {}/{} {:.2f}'.format(i, len(siteNoLst), time.time() - t0), end='\r')
