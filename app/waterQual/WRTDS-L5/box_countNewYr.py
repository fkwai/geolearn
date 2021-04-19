from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# load WRTDS results
dirRoot1 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly')
dirRoot2 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly_rmq')

code = '00955'
dfRes1 = pd.read_csv(os.path.join(dirRoot1, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')
dfRes2 = pd.read_csv(os.path.join(dirRoot2, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')


dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
countMatW = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))
codeLst = sorted(usgs.codeLst)
ic = codeLst.index(code)
ny = 3
count = np.sum(countMatW[:, -ny:, ic], axis=1)
nsLst = np.arange(5, 20)*ny
dataBox = list()
dataBox.append(dfRes1[dfRes1['count']>10]['corr'].values)
for j, ns in enumerate(nsLst):
    dataBox.append(dfRes1[count >= ns]['corr'].values)
fig = figplot.boxPlot(dataBox,  figsize=(12, 4), yRange=[0, 1])
fig.show()