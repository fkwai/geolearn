import random
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec


DF = dbBasin.DataFrameBasin('G200')

code = '00955'
indC = DF.varC.index(code)
countC = np.sum(~np.isnan(DF.c[:, :, indC]), axis=0)
countQ = np.sum(~np.isnan(DF.q[:, :, 1]), axis=0)
indS = np.where((countC > 200) & (countQ > 10000))[0]
siteNoLst = [DF.siteNoLst[ind] for ind in indS]
