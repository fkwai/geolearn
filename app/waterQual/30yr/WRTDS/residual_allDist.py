import functools
from threading import Thread
from contextlib import contextmanager
import signal
from scipy.stats._continuous_distns import _distn_names
import scipy
import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy



wqData = waterQuality.DataModelWQ('rbWN5')
siteNoLst = wqData.siteNoLst
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'All')
dirOut = os.path.join(dirWRTDS, 'output')
dirPar = os.path.join(dirWRTDS, 'params')

# read a temp file
saveFile = os.path.join(dirOut, siteNoLst[0])
dfP = pd.read_csv(saveFile, index_col=None).set_index('date')
t = dfP.index
nt = len(dfP.index)
nc = len(usgs.newC)
ns = len(siteNoLst)
matR = np.ndarray([ns, nt, nc])
matC = np.ndarray([ns, nt, nc])

# calculate residual
t0 = time.time()
for kk, siteNo in enumerate(siteNoLst):
    print('{}/{} {:.2f}'.format(
        kk, len(siteNoLst), time.time()-t0))
    saveFile = os.path.join(dirOut, siteNo)
    dfP = pd.read_csv(saveFile, index_col=None).set_index('date')
    dfP.index = pd.to_datetime(dfP.index)
    dfC = waterQuality.readSiteTS(siteNo, varLst=usgs.newC, freq='W')
    matR[kk, :, :] = dfP.values-dfC.values
    matC[kk, :, :] = dfC.values



def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (
                func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']

distLst = _distn_names
# distLst=['laplace']
dfP = pd.DataFrame(index=distLst, columns=codeLst2)
dfS = pd.DataFrame(index=distLst, columns=codeLst2)
importlib.reload(utils)
for distName in distLst:
    for k, code in enumerate(codeLst2):
        t0 = time.time()
        print('calculating {} {}'.format(code, distName))
        siteNoCode = dictSite[code]
        indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
        ic = usgs.newC.index(code)
        data = matR[indS, :, ic]
        x1 = utils.flatData(data)
        x2 = utils.rmExt(x1, p=5)
        dist = getattr(scipy.stats, distName)
        try:
            func = timeout(timeout=10)(dist.fit)
            args = func(x2)
            s, p = scipy.stats.kstest(x2, distName, args=args)
            dfP.loc[distName][code] = p
            dfS.loc[distName][code] = s
        except:
            print('failed')
        print('time cost = {}'.format(time.time()-t0))
dfP.to_csv('resP.csv')
dfS.to_csv('resS.csv')
