from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)

code = '00915'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
outName = '{}-{}-{}-{}'.format(dataName, code, 'ntnS', trainSet)
siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()

siteNo = siteNoLst[0]
dfP = basins.loadSeq(outName, siteNo)[code]
dfO = waterQuality.readSiteTS(siteNo, [code], freq=wqData.freq)[code]

fig = figplot.tsYr(dfP.index.values, [dfP.values, dfO.values])
