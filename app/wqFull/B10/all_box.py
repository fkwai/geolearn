import importlib
from hydroDL import kPath, utils
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn, dbBasin
from hydroDL.post import axplot, figplot
from hydroDL.master import basinFull

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec

dataNameLst = ['brWN5', 'brDN5']
labelLst = ['QFPRT2C', 'QFPT2C', 'FPRT2QC', 'QT2C']
rhoLst = [365, 10]

dataName = 'brDN5'
label = 'QFPRT2C'
rho = 365
outName = '{}-{}-t{}-B10'.format(dataName, label, rho)
dm = dbBasin.DataModelFull(dataName)
master = basinFull.loadMaster(outName)
varY = master['varY']
testSet = 'all'
sd = '1982-01-01'
ed = '2018-12-31'

yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet=testSet, ep=100)
yO, ycO = basinFull.getObs(outName, testSet, DM=dm)
