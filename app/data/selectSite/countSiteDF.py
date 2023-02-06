from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
import json

DF = dbBasin.DataFrameBasin('dbAll')

indC = [DF.varC.index(code) for code in usgs.varC]
matC = ~np.isnan(DF.c[:, :, indC])
matQ = ~np.isnan(DF.q[:, :, 0])

matB = matC * matQ[:, :, None]

matCount = np.sum(matB, axis=0)

# count rm 00010 and 00095
ny = 44
ind = np.where(np.any(matCount[:, 2:] > ny * 5, axis=1))[0]
siteNoLst = [DF.siteNoLst[x] for x in ind]
saveName = 'siteNoLst_79_23'
saveFile = os.path.join(kPath.dirUsgs, 'siteSel', saveName)
with open(saveFile, 'w') as f:
    json.dump(dict(ny5_rmTK=siteNoLst), f, indent=2)

dataName = 'NY5'
DF = dbBasin.DataFrameBasin.new(
    dataName,
    siteNoLst,
    varC=usgs.varC + ['00060', '00065'],
    varF=gridMET.varLst,
    varG=gageII.varLstEx,
    edStr='2023-01-01',
)


ind = np.where(np.any(matCount[:, 2:] > ny * 10, axis=1))[0]
