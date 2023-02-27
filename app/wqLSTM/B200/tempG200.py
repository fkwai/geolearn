
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

outName='G200-QFPRT2C-rmYr5'
DF = dbBasin.DataFrameBasin('G200')
dictP = basinFull.loadMaster(outName)
outFolder = basinFull.nameFolder(outName)

dictVar = {k: dictP[k] for k in ('varX', 'varXC', 'varY', 'varYC')}
DM = dbBasin.DataModelBasin(DF, subset='all', **dictVar)
DM.loadStat(outFolder)
dataTup = DM.getData()


model = basinFull.defineModel(dataTup, dictP)
model = basinFull.loadModelState(outName, 500, model)
