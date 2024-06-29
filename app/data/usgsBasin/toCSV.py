import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib
import matplotlib.gridspec as gridspec


dataName = "G200"
DF = dbBasin.DataFrameBasin(dataName)

outDir = r'C:\Users\geofk\work\temp\csv'
k=0

dfLst=list()

for k,site in enumerate(DF.siteNoLst):
    print(k)
    df1=pd.DataFrame(data=DF.c[:,k,:],columns=DF.varC,index=DF.t)
    df2=pd.DataFrame(data=DF.q[:,k,:],columns=DF.varQ,index=DF.t)
    df3=pd.DataFrame(data=DF.f[:,k,:],columns=DF.varF,index=DF.t)
    dfMerge=pd.concat([df1,df2,df3],axis=1)
    dfMerge.to_csv(os.path.join(outDir,'{}.csv'.format(site)))
    
dfG = gageII.readData(siteNoLst=DF.siteNoLst)
dfG.to_csv(os.path.join(outDir,'static.csv'))

dfMerge.columns