
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib
import hydroDL


DF1 = dbBasin.DataFrameBasin('G200')
DF2 = dbBasin.DataFrameBasin('N200')

np.sum(~np.isnan(DF1.c))
np.sum(~np.isnan(DF2.c))

siteNo = '09508500'
code = '00600'

dfC, dfCF = usgs.readSample(
    siteNo, codeLst=DF1.varC, startDate=DF1.sd, flag=2, csv=True)

indS = DF1.siteNoLst.index(siteNo)
indC = DF1.varC.index(code)


fig, ax = plt.subplots(1, 1)
ax.plot(DF1.t, DF1.c[:, indS, indC], 'r*')
ax.plot(dfC.index, dfC[code].values, 'b*')
ax.plot(DF2.t, DF2.c[:, indS, indC], 'g*')
fig.show()
