from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os
import argparse
import socket
import json

"""
some states are wrong lead to failed download 
and USGS does not require state now
"""


# find sites from gageII
dfSite = gageII.readData(varLst=['STATE'])
siteNoLstAll = dfSite.index.tolist()

# find existing sites
dirQ = os.path.join(kPath.dirRaw, 'USGS', 'streamflow-instant')
siteNoLstExist = os.listdir(dirQ)

# siteNoLst=sorted(set(siteNoLstAll)-set(siteNoLstExist))
siteNoLst=siteNoLstAll
dictErrQ=dict()
t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    state = dfSite.loc[siteNo]['STATE']
    try:
        saveFile = os.path.join(dirQ, siteNo)
        usgs.downloadHourly(siteNo, saveFile)
    except Exception as e:
        dictErrQ[siteNo] = str(e)
        print('error site Q {}'.format(siteNo))

    ns = len(siteNoLst)
    tc = time.time() - t0
    print('site {}/{} time cost {:.2f}'.format(k, ns, tc))

