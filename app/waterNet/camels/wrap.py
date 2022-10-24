
import json
import numpy as np
from numpy import double
import pandas as pd
import os
import hydroDL
from hydroDL.data import camels, usgs, dbBasin
import importlib
import matplotlib.pyplot as plt
from hydroDL.post import axplot
from hydroDL.utils.time import t2dt
from hydroDL.master import basinFull, slurm

importlib.reload(camels)
dfInfo = camels.dfInfo

# wrapup function
optLst = ['nldas', 'daymet', 'maurer']
dataLst = ['camelsN', 'camelsD', 'camelsM']

# for opt, dataName in zip(optLst, dataLst):
#     dbBasin.io.wrapDataCamels(dataName, optF=opt)
#     DF = dbBasin.DataFrameBasin(dataName)
#     DF.saveSubset('B05', sd='1980-01-01', ed='2004-12-31')
#     DF.saveSubset('A05', sd='2005-01-01', ed='2014-12-31')


for dataName in dataLst:
    DF = dbBasin.DataFrameBasin(dataName)
    DF.saveSubset('WY8095', sd='1980-10-01', ed='1995-09-30')
    DF.saveSubset('WY9510', sd='1995-10-01', ed='2010-09-30')

