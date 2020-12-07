import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np


dirCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
fileCode = os.path.join(dirCode, 'basinCode')
dfCode = pd.read_csv(fileCode, dtype={'siteNo': str}).set_index('siteNo')

# assign region code
dfCode['region'] = 'X'
# A - 5
temp = dfCode['code0'] == 5
dfCode.at[temp, 'region'] = 'A'
# B - 6
temp = dfCode['code0'] == 6
dfCode.at[temp, 'region'] = 'B'
# C - 7
temp = dfCode['code0'] == 7
dfCode.at[temp, 'region'] = 'C'
# D - 8.1
temp = (dfCode['code0'] == 8) & (dfCode['code1'] == 1)
dfCode.at[temp, 'region'] = 'D'
# E - 8.2
temp = (dfCode['code0'] == 8) & (dfCode['code1'] == 2)
dfCode.at[temp, 'region'] = 'E'
# F - 8.3
temp = (dfCode['code0'] == 8) & (dfCode['code1'] == 3)
dfCode.at[temp, 'region'] = 'F'
# G - 8.4
temp = (dfCode['code0'] == 8) & (dfCode['code1'] == 4)
dfCode.at[temp, 'region'] = 'G'
# H - 8.5
temp = (dfCode['code0'] == 8) & (dfCode['code1'] == 5)
dfCode.at[temp, 'region'] = 'H'
# I - 9.2
temp = (dfCode['code0'] == 9) & (dfCode['code1'] == 2)
dfCode.at[temp, 'region'] = 'I'
# J - 9.3
temp = (dfCode['code0'] == 9) & (dfCode['code1'] == 3)
dfCode.at[temp, 'region'] = 'J'
# K - 9.4
temp = (dfCode['code0'] == 9) & (dfCode['code1'] == 4)
dfCode.at[temp, 'region'] = 'K'
# L - 9.5, 9.6
temp = (dfCode['code0'] == 9) & (
    (dfCode['code1'] == 5) | (dfCode['code1'] == 6))
dfCode.at[temp, 'region'] = 'L'
# M - 10.1
temp = (dfCode['code0'] == 10) & (dfCode['code1'] == 1)
dfCode.at[temp, 'region'] = 'M'
# M - 10.2
temp = (dfCode['code0'] == 10) & (dfCode['code1'] == 2)
dfCode.at[temp, 'region'] = 'N'
# O - 11.1
temp = (dfCode['code0'] == 11) & (dfCode['code1'] == 1)
dfCode.at[temp, 'region'] = 'O'
# P - 12.1
temp = (dfCode['code0'] == 12) & (dfCode['code1'] == 1)
dfCode.at[temp, 'region'] = 'P'
# Q - 13
temp = dfCode['code0'] == 13
dfCode.at[temp, 'region'] = 'Q'
# R - 14.3, 15.4
temp = ((dfCode['code0'] == 14) & (dfCode['code1'] == 3)) | (
    (dfCode['code0'] == 15) & (dfCode['code1'] == 4))
dfCode.at[temp, 'region'] = 'R'

dfCode.to_csv(os.path.join(dirCode, 'basinRegion'))

dfCode['region'].value_counts()