
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.app.waterQuality import WRTDS, wqRela
import statsmodels.api as sm
import scipy
from hydroDL.app.waterQuality import cqType
import importlib
import time
import statsmodels.api as sm
from hydroDL.data import dbBasin, gageII, usgs
from hydroDL.master import basinFull


dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)


sn = 1e-5
thP = 0.0001
thR = 0.6

code = '00915'
[matK, matN, matA, matB, matP, matR] = [
    np.full(len(DF.siteNoLst), np.nan) for x in range(6)]
# siteNo = '06800000'

dataName = 'G200'
# DF = dbBasin.DataFrameBasin(dataName)

sn = 1e-5
thP = 0.0001
thR = 0.5

code = '00915'
siteNo = '06800000'
siteNo = DF.siteNoLst[10]

indS = DF.siteNoLst.index(siteNo)
indC = DF.varC.index(code)
Q = DF.q[:, indS, 1]
C = DF.c[:, indS, indC]
[x, y], _ = utils.rmNan([Q, C])
Qs = np.sort(Q[~np.isnan(Q)])
ceq, dw, out = wqRela.kateModel(x, y, Qs)
ceq/(1+x/dw)
# plot
fig, ax = plt.subplots(1, 1)
ax.plot(np.log(x+sn), np.log(y+sn), 'k*')
ax.plot(np.log(Qs+sn), np.log(out+sn), '-r')
fig.show()
