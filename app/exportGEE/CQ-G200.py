import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import sklearn.tree
import matplotlib.gridspec as gridspec


dirGEE = r'C:\Users\geofk\work\GEE'
outName = 'CQ-G200'
dirOut = os.path.join(dirGEE, outName)
if ~ os.path.exists(dirOut):
    os.makedirs(dirOut)

DF = dbBasin.DataFrameBasin('G200')

lat, lon = DF.getGeo()
siteNoLst = DF.siteNoLst
dfCrd = pd.DataFrame(index=siteNoLst, columns=['lat', 'lon'])
dfCrd['lat'] = lat
dfCrd['lon'] = lon
dfCrd.index.name = 'siteNo'
dfCrd.to_csv(os.path.join(dirOut, 'crd.csv'))
