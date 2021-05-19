
from hydroDL.data import dbBasin
import json
import os
from hydroDL import kPath

sd = '1982-01-01'
ed = '2018-12-31'
dictSiteName = 'dictRB_Y28N5.json'
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
freqLst = ['W', 'D']
for freq in freqLst:
    dataName = 'bs'+freq+'N'+str(5)
    siteNoLst = dictSite['comb']
    DM = dbBasin.DataModelFull.new(
        dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)

    dataName = 'br'+freq+'N'+str(5)
    siteNoLst = dictSite['rmTK']
    DM = dbBasin.DataModelFull.new(
        dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)

# dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
# dictSiteName = 'dictWeathering.json'
# with open(os.path.join(dirSel, dictSiteName)) as f:
#     dictSite = json.load(f)
# siteNoLst = dictSite['k12']

# sd = '1982-01-01'
# ed = '2018-12-31'
# dataName = 'weathering'
# freq = 'D'
# DM = dbBasin.DataModelFull.new(
#     dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)