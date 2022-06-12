
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

dataName = 'QN90ref'
DF = dbBasin.DataFrameBasin(dataName)
f = DF.f
c = DF.c
q = DF.q[:, :, 0]
t = DF.t
siteNo = DF.siteNoLst
dfG = gageII.readData(siteNoLst=DF.siteNoLst)
folder = r'C:\Users\geofk\work\waterQuality\tempData\Zach'
dfG.to_csv(os.path.join(folder, 'geoConst.csv'))
np.savez_compressed(os.path.join(folder, 'data'),
                    f=f, varF=DF.varF,
                    c=c, varC=DF.varC,
                    q=q, t=t, siteNo=siteNo)

dfCrd = dfG[['LAT_GAGE', 'LNG_GAGE']]
dfCrd.to_csv(os.path.join(folder, 'basinIdCrd.csv'))
