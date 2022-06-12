
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
f = DF.f
c = DF.c
q = DF.q[:, :, 0]
t = DF.t
siteNo = DF.siteNoLst
dfG = gageII.readData(siteNoLst=DF.siteNoLst)
folder = r'C:\Users\geofk\work\waterQuality\tempData\Lijing'
dfG.to_csv(os.path.join(folder, 'geoConst'))
np.savez_compressed(os.path.join('data'),
                    f=f, varF=DF.varF,
                    c=c, varC=DF.varC,
                    q=q, t=t, siteNo=siteNo)

