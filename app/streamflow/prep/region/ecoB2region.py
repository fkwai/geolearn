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
fileCode = os.path.join(dirCode, 'basinEcoB')
dfCode = pd.read_csv(fileCode, dtype={'siteNo': str}).set_index('siteNo')

# assign region code
dfCode['region'] = 'X'
# A - 5
temp = dfCode['EcoB1'] == 5
dfCode.at[temp, 'region'] = 'A'
# B - 6
temp = dfCode['EcoB1'] == 6
dfCode.at[temp, 'region'] = 'B'
# C - 7
temp = dfCode['EcoB1'] == 7
dfCode.at[temp, 'region'] = 'C'
# D - 8.1
temp = (dfCode['EcoB1'] == 8) & (dfCode['EcoB2'] == 1)
dfCode.at[temp, 'region'] = 'D'
# E - 8.2
temp = (dfCode['EcoB1'] == 8) & (dfCode['EcoB2'] == 2)
dfCode.at[temp, 'region'] = 'E'
# F - 8.3
temp = (dfCode['EcoB1'] == 8) & (dfCode['EcoB2'] == 3)
dfCode.at[temp, 'region'] = 'F'
# G - 8.4
temp = (dfCode['EcoB1'] == 8) & (dfCode['EcoB2'] == 4)
dfCode.at[temp, 'region'] = 'G'
# H - 8.5
temp = (dfCode['EcoB1'] == 8) & (dfCode['EcoB2'] == 5)
dfCode.at[temp, 'region'] = 'H'
# I - 9.2
temp = (dfCode['EcoB1'] == 9) & (dfCode['EcoB2'] == 2)
dfCode.at[temp, 'region'] = 'I'
# J - 9.3
temp = (dfCode['EcoB1'] == 9) & (dfCode['EcoB2'] == 3)
dfCode.at[temp, 'region'] = 'J'
# K - 9.4
temp = (dfCode['EcoB1'] == 9) & (dfCode['EcoB2'] == 4)
dfCode.at[temp, 'region'] = 'K'
# L - 9.5, 9.6
temp = (dfCode['EcoB1'] == 9) & (
    (dfCode['EcoB2'] == 5) | (dfCode['EcoB2'] == 6))
dfCode.at[temp, 'region'] = 'L'
# M - 10.1
temp = (dfCode['EcoB1'] == 10) & (dfCode['EcoB2'] == 1)
dfCode.at[temp, 'region'] = 'M'
# M - 10.2
temp = (dfCode['EcoB1'] == 10) & (dfCode['EcoB2'] == 2)
dfCode.at[temp, 'region'] = 'N'
# O - 11.1
temp = (dfCode['EcoB1'] == 11) & (dfCode['EcoB2'] == 1)
dfCode.at[temp, 'region'] = 'O'
# P - 12.1
temp = (dfCode['EcoB1'] == 12) & (dfCode['EcoB2'] == 1)
dfCode.at[temp, 'region'] = 'P'
# Q - 13
temp = dfCode['EcoB1'] == 13
dfCode.at[temp, 'region'] = 'Q'
# R - 14.3, 15.4
temp = ((dfCode['EcoB1'] == 14) & (dfCode['EcoB2'] == 3)) | (
    (dfCode['EcoB1'] == 15) & (dfCode['EcoB2'] == 4))
dfCode.at[temp, 'region'] = 'R'

dfCode.to_csv(os.path.join(dirCode, 'basinRegionB'))

dfCode['region'].value_counts()
