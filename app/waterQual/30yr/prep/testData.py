from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import time
import json

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')

# countMat = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMat = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))


codeSel = ['00915', '00955']

# count for obs before / after 2010
count1 = np.ndarray([len(siteNoLstAll), len(codeSel)])
count2 = np.ndarray([len(siteNoLstAll), len(codeSel)])
for ic, code in enumerate(codeSel):
    iCode = codeLst.index(code)
    count1[:, ic] = np.sum(countMat[:, :30, iCode], axis=1)
    count2[:, ic] = np.sum(countMat[:, 30:, iCode], axis=1)

# get siteNoLst
ns = 10
pickMat = (count1 >= ns*30) & (count2 >= ns*10)
siteNoLst = list(np.array(siteNoLstAll)[np.any(pickMat, axis=1)])
len(siteNoLst)

# save for each code and comb
dictSite = dict()
indS = np.where(np.any(pickMat, axis=1))[0]
dictSite['comb'] = [siteNoLstAll[ind] for ind in indS]
for code in codeSel:
    ic = codeSel.index(code)
    indS = np.where(pickMat[:, ic])[0]
    dictSite[code] = [siteNoLstAll[ind] for ind in indS]
saveName = os.path.join(dirInv, 'siteSel', 'test')
with open(saveName+'.json', 'w') as fp:
    json.dump(dictSite, fp, indent=4)

# wrap up
dataName = 'test'
with open(saveName+'.csv') as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
freq = 'W'
rho = 365 if freq == 'D' else 52
wqData = waterQuality.DataModelWQ.new(
    dataName, siteNoLst, rho=rho, freq=freq)

# subset
wqData = waterQuality.DataModelWQ(dataName)
info = wqData.info
info['yr'] = pd.DatetimeIndex(info['date']).year
for code in codeSel+['comb']:
    print(dataName, code)
    siteNoLst = dictSite[code]
    bs = info['siteNo'].isin(siteNoLst)
    b1 = (info['yr'] < 2010).values
    b2 = (info['yr'] >= 2010).values
    if code == 'comb':
        ind1 = info.index[b1 & bs].values
        ind2 = info.index[b2 & bs].values
    else:
        if len(wqData.c.shape) == 2:
            bv = ~np.isnan(wqData.c[:, wqData.varC.index(code)])
        elif len(wqData.c.shape) == 3:
            bv = ~np.isnan(wqData.c[-1, :, wqData.varC.index(code)])
        ind1 = info.index[b1 & bs & bv].values
        ind2 = info.index[b2 & bs & bv].values
    wqData.saveSubset('{}-B10'.format(code), ind1)
    wqData.saveSubset('{}-A10'.format(code), ind2)
