from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

DF = dbBasin.DataFrameBasin('G200')

# predefine
t1 = np.datetime64('1982-01-01')
t2 = np.datetime64('2010-01-01')
t3 = np.datetime64('2018-12-31')
indT1 = np.where(DF.t == t1)[0][0]
indT2 = np.where(DF.t == t2)[0][0]
indT3 = np.where(DF.t == t3)[0][0]

# count for code
code = '00600'
codeLst = ['00600', '00618', '00915', '00945', '00955']
pLst = [100, 75, 50, 25]
nyLst = [6, 8, 10]

for code in codeLst:
    indC = DF.varC.index(code)
    temp = ~np.isnan(DF.q[:, :, 0]) & ~np.isnan(DF.c[:, :, indC])
    count1 = np.sum(temp[indT1:indT2+1], axis=0)
    count2 = np.sum(temp[indT2:indT3+1], axis=0)
    np.sort(count1)
    np.sort(count2)
    # select sites
    for ny in nyLst:
        th1 = ny*30
        th2 = ny*10
        indSel = np.where((count1 > th1) & (count2 > th2))[0]
        len(indSel)
        siteNoLst = [DF.siteNoLst[x] for x in indSel]
        # subset
        for p in pLst:
            mask = np.ones([indT2-indT1+1, len(siteNoLst)]).astype(bool)
            for k, siteNo in enumerate(siteNoLst):
                indS = DF.siteNoLst.index(siteNo)
                temp = np.where(~np.isnan(DF.c[:, indS, indC]))[0]
                indT = temp[temp <= indT2]
                np.random.seed(int(siteNo[:10]))
                np.random.shuffle(indT)
                mask[indT[:int(len(indT)*p/100)], k] = False
            subName = '{}-n{}-p{}-B10'.format(code, ny, p)
            DF.saveSubset(subName, sd=str(t1), ed=str(t2),
                          siteNoLst=siteNoLst, mask=mask)
        testSet = '{}-n{}-A10'.format(code, ny)
        DF.saveSubset(testSet, sd=str(t2), ed=str(t3),
                      siteNoLst=siteNoLst)

# train
