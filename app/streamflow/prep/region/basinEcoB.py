import shapefile
from shapely.geometry import Point, shape
import hydroDL
from hydroDL.data import dbCsv
from hydroDL.utils import gis, grid
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL import kPath
import time
import csv
import os
import pandas as pd
import numpy as np


# load sites
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
varLst = ['ECO3_BAS_DOM', 'LAT_GAGE', 'LNG_GAGE', 'CLASS']
dfR = gageII.readData(varLst=varLst, siteNoLst=siteNoLst)
dfR = gageII.updateCode(dfR)
fileT = os.path.join(gageII.dirTab, 'lookupEco.csv')
tabT = pd.read_csv(fileT).set_index('Eco3code')

mat = np.full([len(siteNoLst), 3], np.nan)
for code in range(1, 85):
    siteNoTemp = dfR[dfR['ECO3_BAS_DOM'] == code].index
    ind = [siteNoLst.index(siteNo) for siteNo in siteNoTemp]
    eco3 = tabT.loc[code]['Eco3']
    EcoB1, EcoB2, EcoB3 = eco3.split('.')
    mat[ind, 0] = EcoB1
    mat[ind, 1] = EcoB2
    mat[ind, 2] = EcoB3
dfEcoB = pd.DataFrame(index=siteNoLst, columns=[
                      'EcoB1', 'EcoB2', 'EcoB3'], data=mat)
dfEcoB.index.name = 'siteNo'
dfEcoB.to_csv(os.path.join(dirInv, 'ecoregion', 'basinEcoB'))


# dirCode = os.path.join(kPath.dirData, 'USGS', 'inventory', 'ecoregion')
# fileCode = os.path.join(dirCode, 'basinEco')
# dfCode = pd.read_csv(fileCode, dtype={'siteNo': str}).set_index('siteNo')
