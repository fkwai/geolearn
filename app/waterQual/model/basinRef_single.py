from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('basinRef')

doLst = list()
doLst.append('subset')

if 'subset' in doLst:
    # find ind have SiO4, NO3
    codeLst = ['00618', '00955']
    icLst = [wqData.varC.index(code) for code in codeLst]
    indAll = np.where(~np.isnan(wqData.c[:, icLst]).all(axis=1))[0]
    indAny = np.where(~np.isnan(wqData.c[:, icLst]).any(axis=1))[0]
    # print number of samples
    for code in codeLst:
        ic = wqData.varC.index(code)
        indC = np.where(~np.isnan(wqData.c[:, ic]))[0]
    # seperate index by years
    for ind, lab in zip([indAll, indAny], ['all', 'any']):
        indYr = waterQuality.indYr(
            wqData.info.iloc[ind], yrLst=[1979, 2000])[0]
        indYrCmp = np.setdiff1d(ind, indYr)
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y8090']), indYr)
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'rmY8090']), indYrCmp)
    for code in codeLst:
        ic = wqData.varC.index(code)
        indC = np.where(~np.isnan(wqData.c[:, ic]))[0]
        indYr = waterQuality.indYr(
            wqData.info.iloc[indC], yrLst=[1979, 2000])[0]
        indYrCmp = np.setdiff1d(indC, indYr)
        wqData.saveSubset(code+'-Y8090', indYr)
        wqData.saveSubset(code+'-rmY8090', indYrCmp)
    # d=wqData.info.iloc[wqData.subset['00618-00955-any-Y10']]['date']
    # np.sort(pd.DatetimeIndex(d).year.unique())
    # ind=wqData.info.iloc[wqData.subset['00618-00955-any-Y10']].index.values
    # wqData.c[ind, wqData.varC.index('00618')]

# train local

