

import matplotlib.gridspec as gridspec
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib

DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC

# count
trainSet = 'rmYr5'
testSet = 'pkYr5'
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)

# load linear/seasonal
dirParLst = [r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param',
             r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\Q\param',
             r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\S\param']
saveNameLst = ['QS', 'Q', 'S']
for dirPar, saveName in zip(dirParLst, saveNameLst):
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirPar, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan

    # plot map
    lat, lon = DF.getGeo()
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(5, 4)
    for k, code in enumerate(codeLst):
        j, i = utils.index2d(k, 5, 4)
        ax = mapplot.mapPoint(fig, gs[j:j+1, i:i+1], lat, lon,
                              matLR[:, k], cb=True)
        codeStr = usgs.codePdf.loc[code]['shortName']
        ax.set_title('{} {}'.format(code, codeStr))
    plt.tight_layout()
    fig.show()
    dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
    plt.savefig(os.path.join(dirPaper, 'mapSim_{}'.format(saveName)))
