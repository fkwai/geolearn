import scipy
from scipy.stats import shapiro,kstest,kurtosis,skew
import random
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
import statsmodels.api as sm
import time

dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)
siteNoLst = DF.siteNoLst
codeLst = DF.varC

matk1,matk2,mats1,mats2=[np.full([len(siteNoLst),len(codeLst)],np.nan) for k in range(4)]

for var in DF.varC:
    for siteNo in DF.siteNoLst:
        indC=DF.varC.index(var)
        indS=DF.siteNoLst.index(siteNo)
        data = DF.c[:,indS,indC]
        x = data[~np.isnan(data)]    
        logx = np.log(x+1e-5)
        if len(x)>200:
            mats1[indS,indC]= skew(x)
            mats2[indS,indC]= skew(logx)
            matk1[indS,indC]= kurtosis(x)
            matk2[indS,indC]= kurtosis(logx)   

# skew
codeStrLst = list()
dataPlotK = list()
dataPlotS = list()
for code in DF.varC: 
    k=DF.varC.index(code)   
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    dataPlotK.append([matk1[:, k], matk2[:, k]])
    dataPlotS.append([mats1[:, k], mats2[:, k]])
fig, axes = figplot.boxPlot(
    dataPlotK, widths=0.5, figsize=(12, 4), label1=codeStrLst)
plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
fig.show()

fig, axes = figplot.boxPlot(
    dataPlotS, widths=0.5, figsize=(12, 4), label1=codeStrLst)
for ax in axes:
    ax.axhline(0, color='k')
plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
fig.show()

