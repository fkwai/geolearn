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
from scipy.stats import shapiro,kstest,kurtosis,skew


DF = dbBasin.DataFrameBasin('rmTK-B200')
mat=DF.c
matB=~np.isnan(mat)
count = np.nansum(matB, axis=0)

sLst=list()
kLst=list()
the=100
for code in usgs.varC:
    indC=usgs.varC.index(code)
    data=mat[:,:,indC]
    k1=kurtosis(data,nan_policy='omit')
    k2=kurtosis(np.log(data),nan_policy='omit')
    k1[count<the]=np.nan
    k2[count<the]=np.nan
    s1=skew(data,nan_policy='omit')
    s2=skew(np.log(data),nan_policy='omit')
    s1[count<the]=np.nan
    s2[count<the]=np.nan
    sLst.append([s1,s2])
    kLst.append([k1,k2])
