import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy


# WRTDS corr
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-D', 'All')
fileC = os.path.join(dirWrtds, 'corr')
dfCorr = pd.read_csv(fileC, dtype={'siteNo': str}).set_index('siteNo')

code = '00915'
codeName = usgs.codePdf.loc[code]['shortName']
# load WRTDS par
fileP = os.path.join(dirWrtds, 'params', code)
dfPar = pd.read_csv(fileP, dtype={'siteNo': str}).set_index('siteNo')
# select site by count
n = 40*2
dfParSel = dfPar[dfPar['count'] > n]
siteNoLst = dfParSel.index.tolist()
dfCorrSel = dfCorr.loc[siteNoLst][code]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
dfCrd = gageII.updateCode(dfCrd)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values

# plot map
parLst = ['pQ', 'pSinT', 'pCosT', 'pYr', 'b']
figM, axM = plt.subplots(3, 2, figsize=(12, 16))
axplot.mapPoint(axM[0, 0], lat, lon, dfCorrSel.values, s=16)
axM[0, 0].set_title('WRTDS corr {}'.format(codeName))
for k, par in enumerate(parLst):
    iy, ix = utils.index2d(k+1, 3, 2)
    axplot.mapPoint(axM[iy, ix], lat, lon, dfParSel[par].values, s=16)
    axM[iy, ix].set_title('WRTDS {} {}'.format(par, codeName))
figM.show()

