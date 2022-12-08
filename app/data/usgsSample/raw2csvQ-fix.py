
from hydroDL.data import usgs, gageII
from hydroDL import kPath
from hydroDL.app import waterQuality
import pandas as pd
import numpy as np
import time
import os

dirRaw = os.path.join(kPath.dirRaw, 'USGS', 'streamflow')
dirCsv = os.path.join(kPath.dirUSGS, 'streamflow','csv')

rawLst = [f for f in sorted(os.listdir(dirRaw))]
csvLst = [f for f in sorted(os.listdir(dirCsv))]
siteNoLst = [f for f in rawLst if f not in csvLst]


# read all C/Q data and save as csv - improve future efficiency
# streamflow
t0 = time.time()
for i, siteNo in enumerate(siteNoLst):
    dfQ = usgs.readStreamflow(siteNo,csv=False)
    if dfQ is not None:
        dfQ.to_csv(os.path.join(dirCsv, siteNo))
    print('\t {}/{} {:.2f}'.format(
        i, len(siteNoLst), time.time()-t0), end='\r')
