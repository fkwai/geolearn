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

wqData = waterQuality.DataModelWQ('HBN')
figFolder = os.path.join(kPath.dirWQ, 'HBN')

doLst = list()
doLst.append('subset')

if 'subset' in doLst:
    # find ind have SiO4, P
    codeLst = ['00618', '00950'，'00681']
    ic = [wqData.varC.index(code) for code in codeLst]
    indAll = np.where(~np.isnan(wqData.c[:, ic]).all(axis=1))[0]
    # seperate index by years
    indYr = waterQuality.indYr(wqData.info.iloc[indAll], yrLst=[2010, 2020])[0]
    indYrCmp = np.setdiff1d(indAll, indYr)
    wqData.saveSubset('-'.join(sorted(codeLst))+'-Y10', indYr)
    wqData.saveSubset('-'.join(sorted(codeLst))+'-rmY10', indYrCmp)
    for code in codeLst:
        ic = wqData.varC.index(code)
        indC = np.where(~np.isnan(wqData.c[:, ic]))[0]
        indYr = waterQuality.indYr(
            wqData.info.iloc[indC], yrLst=[2010, 2020])[0]
        indYrCmp = np.setdiff1d(indC, indYr)
        wqData.saveSubset(code+'-Y10', indYr)
        wqData.saveSubset(code+'-rmY10', indYrCmp)
    # d=wqData.info.iloc[wqData.subset['00665-Y10']]['date']
    # np.sort(pd.DatetimeIndex(d).year.unique())
    # ind=wqData.info.iloc[wqData.subset['00665-Y10']].index.values
    # wqData.c[ind, wqData.varC.index('00665')]

