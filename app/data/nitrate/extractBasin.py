
import shapefile
import random
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec

DF = dbBasin.DataFrameBasin('G200')
latA, lonA = DF.getGeo()

code = '00600'
ic = DF.varC.index(code)
countC = np.sum(~np.isnan(DF.c[:, :, ic]), axis=0)
indS = np.where(countC >= 200)[0]

C = DF.c[:, indS, ic]
Q = DF.q[:, indS, 1]

lat = latA[indS]
lon = lonA[indS]
siteNoSel = [DF.siteNoLst[x] for x in indS]
matR = np.nanmean(C, axis=0)

shpFile = os.path.join(kPath.dirData, 'USGS',
                       'basins', 'basinAll_prj.shp')
sf = shapefile.Reader(shpFile)
shapeLst = sf.shapes()
recLst = sf.records()
siteNoLst = [rec[2] for rec in recLst]

outShapeFile = r'C:\Users\geofk\work\map\usgs\nitrate\{}_G200.shp'.format(code)

with shapefile.Writer(outShapeFile) as w:
    w.fields = sf.fields[1:]  # skip first deletion field
    for siteNo in siteNoSel:
        print(siteNo)
        ind = siteNoLst.index(siteNo)
        w.record(*[recLst[ind][x] for x in range(len(w.fields))])
        w.shape(shapeLst[ind])


[recLst[ind][x] for x in range(len(w.fields))]