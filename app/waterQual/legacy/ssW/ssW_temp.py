from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataName = 'ssW'
wqData = waterQuality.DataModelWQ(dataName)

code = '00940'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)

ind = wqData.subset[trainSet]
indC = wqData.varC.index(code)
aa = wqData.c[ind, indC]
len(np.where(~np.isnan(aa))[0])
