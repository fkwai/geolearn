from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os
import argparse
import socket
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-iS', dest='iS', type=int, default=0)
    parser.add_argument('-iE', dest='iE', type=int, default=9067)
    args = parser.parse_args()
    iS = args.iS
    iE = args.iE

print(iS)
print(iE)
socket.setdefaulttimeout(30)

# find sites from gageII
dfSite = gageII.readData(varLst=['STATE'])
siteNoLstAll = dfSite.index.tolist()
siteNoLst = siteNoLstAll[iS:iE]

dictErrC = dict()
dictErrQ = dict()

t0 = time.time()
for k, siteNo in enumerate(siteNoLst):
    state = dfSite.loc[siteNo]['STATE']
    try:
        saveFile = os.path.join(kPath.dirRaw, 'USGS', 'streamflow', siteNo)
        usgs.downloadDaily(siteNo, ['00060'], saveFile, state)
    except Exception as e:
        dictErrQ[siteNo] = str(e)
        print('error site Q {}'.format(siteNo))
    time.sleep(1)

    try:
        saveFile = os.path.join(kPath.dirRaw, 'USGS', 'sample', siteNo)
        usgs.downloadSample(siteNo, saveFile, state)
    except Exception as e:
        dictErrC[siteNo] = str(e)
        print('error site C {}'.format(siteNo))
    time.sleep(1)

    ns = len(siteNoLst)
    tc = time.time() - t0
    print('site {}/{} time cost {:.2f}'.format(k, ns, tc))

errFileQ = os.path.join(kPath.dirRaw, 'USGS', 'errQ-{}-{}.csv'.format(iS, iE))
with open(errFileQ + '.json', 'w') as f:
    json.dump(dictErrQ, f)
errFileC = os.path.join(kPath.dirRaw, 'USGS', 'errC-{}-{}.csv'.format(iS, iE))
with open(errFileC + '.json', 'w') as f:
    json.dump(dictErrC, f)
