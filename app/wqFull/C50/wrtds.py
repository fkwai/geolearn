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


DF = dbBasin.DataFrameBasin('weathering')
trainSet = 'rmRT20'
testSet = 'pkRT20'
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']

# Calculate WRTDS from train and test set
varX = ['00060', 'sinT', 'cosT', 'datenum']
varY = codeSel
DM = dbBasin.DataModelBasin(DF, subset=trainSet, varX=varX, varY=varY)

###
X = DM.X
Y = DM.Y
T = DM.t

k = 100
dY = np.abs(T[ind]-T).values)
dQ=np.abs((df2.loc[t]['logQ']-df1['logQ']).values)
dS=np.min(
    np.stack([abs(np.ceil(dY)-dY), abs(dY-np.floor(dY))]), axis = 0)
d=np.stack([dY, dQ, dS])
