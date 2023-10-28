import hydroDL.data.usgs.read as read
import os
import hydroDL.kPath as kPath
import pandas as pd
from hydroDL.data import usgs


dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileG = os.path.join(dirInv, 'inv-gageII')
fileQ = os.path.join(dirInv, 'inv-surfaceWater')
fileC = os.path.join(dirInv, 'inv-waterQuality')
tabG = read.readUsgsText(fileG)
tabQ = read.readUsgsText(fileQ)
tabC = read.readUsgsText(fileC)
sG = tabG['site_no'].tolist()
sQ = tabQ['site_no'].tolist()
sC = tabC['site_no'].tolist()
siteNoSet=set(sG).intersection(set(sC)).intersection(set(sQ))
lstInv=sorted(list(siteNoSet))

# sites in gageII but not streamflow - turn out to be 
# inventory is only a reference. Downloaded all 9067 gageII sites and check
ss=set(sG)-set(sQ)
aa=set(ss).intersection(set(sC))
len(aa)


seriesSum=pd.Series()
for k, siteNo in enumerate(siteNoLst):
    k
    dfC = usgs.readSample(siteNo, startDate='1982-01-01')
    temp=(~dfC.isna()).sum(axis=0)
    seriesSum=seriesSum.add(temp,fill_value=0)

