
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

siteNo = '09163500'
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']
varF = ['pr', 'etr', 'tmmn', 'srad', 'LAI']
indS = DF.siteNoLst.index(siteNo)
indC = [DF.varC.index(code) for code in codeLst]
indF = [DF.varF.index(var) for var in varF]

C = DF.c[:, indS, indC]
Q = DF.q[:, indS, 1]/365*1000
F = DF.f[:, indS, indF]


labelLst = list()
for k, code in enumerate(codeLst):
    labelLst.append('{}'.format(usgs.codePdf.loc[code]['shortName']))
fig, axes = figplot.multiTS(DF.t, C, labelLst=labelLst)
fig.show()

labelLst = ['Prcp', 'Evp', 'Temp', 'Rad', 'LAI']
fig, axes = figplot.multiTS(DF.t, F, labelLst=labelLst, cLst='b')
fig.show()

fig, ax = plt.subplots(1, 1,  figsize=(12, 3))
axplot.plotTS(ax, DF.t, Q, cLst='r')
fig.show()

fig, ax = plt.subplots(1, 1,  figsize=(12, 3))
axplot.plotTS(ax, DF.t,  DF.f[:, indS, 0], cLst='b')
fig.show()