from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import scipy

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
yrLst = list(range(1980, 2020))

# countMatD = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMatW = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))

# num of sample each year
groupLst = [
    ['00010', '00095', '00400', '70303', '80154'],
    ['00600', '00605', '00618', '00660', '00665', '00681', '71846'],
    ['00915', '00925', '00930', '00935', '00940', '00945', '00950', '00955'],
    ['00300', '00405', '00440', '00410']]
codePdf = usgs.codePdf
for group in groupLst:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for code in group:
        ic = codeLst.index(code)
        label = '{} {}'.format(code, codePdf.loc[code]['shortName'])
        count = countMatW[:, :, ic].copy()
        indS = np.where((np.sum(count > 10, axis=1) > 6) |
                        np.sum(count > 20, axis=1) > 2)[0]
        countYr = np.sum(count[indS, :], axis=0)
        # countYr = np.sum(count, axis=0)
        ax.plot(yrLst, countYr, '-*', label=label)
    ax.legend()
    fig.show()
