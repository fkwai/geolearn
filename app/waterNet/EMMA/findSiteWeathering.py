import random
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec


DF = dbBasin.DataFrameBasin('G200')
usgs.codePdf

codeLst = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
nc = len(codeLst)
indC = [DF.varC.index(code) for code in codeLst]
countC = np.sum(~np.isnan(DF.c[:, :, indC]), axis=0)
countQ = np.sum(~np.isnan(DF.q[:, :, 1]), axis=0)

# any
indS = np.where((countC > 200).any(axis=1) & (countQ > 10000))[0]
siteNoLst = [DF.siteNoLst[ind] for ind in indS]
dataName = 'weaG200'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=codeLst, varG=gageII.varLstEx)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')


# any
indS = np.where((countC > 200).any(axis=1) & (countQ > 10000))[0]
siteNoLst = [DF.siteNoLst[ind] for ind in indS]
dataName = 'weaG200All'
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=codeLst, varG=gageII.varLstEx)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')