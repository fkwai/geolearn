# read inventory of all sites
from hydroDL.data import usgs, gageII
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# read site inventory
workDir = r'C:\Users\geofk\work\waterQuality'
modelDir = os.path.join(workDir, 'modelUsgs2')
fileInvC = os.path.join(workDir, 'inventory_NWIS_sample')
fileInvQ = os.path.join(workDir, 'inventory_NWIS_streamflow')

# look up sample for interested sample sites
fileCountC = os.path.join(workDir, 'count_NWIS_sample')
if os.path.exists(fileCountC):
    tabC = pd.read_csv(fileCountC, dtype={'site_no': str})
else:
    siteC = usgs.readUsgsText(fileInvC)
    codeLst = \
        ['00915', '00925', '00930', '00935', '00955', '00940', '00945']+\
        ['00418','00419','39086','39087']+\
        ['00301','00300','00618','00681','00653']+\
        ['00010','00530','00094']+\
        ['00403','00408']
    dictTab = dict()
    for code in codeLst:
        site = siteC.loc[(siteC['parm_cd'] == code) & (siteC['count_nu'] > 1)]
        temp = dict(
            zip(site['site_no'].tolist(),
                site['count_nu'].astype(int).tolist()))
        dictTab[code] = temp
    tabC = pd.DataFrame.from_dict(dictTab)
    tabC = tabC.rename_axis('site_no').reset_index()
    tabC.to_csv(fileCountC, index=False)

# screen site
siteQ = usgs.readUsgsText(os.path.join(workDir, 'inventory_NWIS_streamflow'))
tabGageII = gageII.readTab('bas_classif')
idLstC = tabC.loc[tabC.sum(axis=1) > 300]['site_no'].tolist()
idLstQ = siteQ['site_no'].tolist()
idLstG = tabGageII['STAID'].tolist()
siteNoLst = list(
    set(idLstC).intersection(set(idLstQ)).intersection(set(idLstG)))

# download C/Q data
errLst = list()
tabState = pd.read_csv(os.path.join(workDir, 'fips_state_code.csv'))
for siteNo in siteNoLst:    
    try:
        stateCd = siteQ['state_cd'].loc[siteQ['site_no'] == siteNo].values[0]
        state = tabState['short'].loc[tabState['code'] == int(
            stateCd)].values[0]
        saveFile = os.path.join(workDir, 'USGS', 'dailyTS', siteNo)
        if not os.path.exists(saveFile):
            print('Q: '+ siteNo)
            usgs.downloadDaily(siteNo, ['00060'], state, saveFile)
        saveFile = os.path.join(workDir, 'USGS', 'sample', siteNo)
        if not os.path.exists(saveFile):
            print('C: '+ siteNo)
            usgs.downloadSample(siteNo, state, saveFile)
    except:
        errLst.append(siteNo)
# ['08015500', '02245500', '07093700', '01651800']


# download forcing
## wrap up a csv for sites
sdLst = list()
edLst = list()
tempLst = list()
for siteNo in siteNoLst:
    fileName = os.path.join(workDir, 'USGS', 'dailyTS', siteNo)
    data = usgs.readUsgsText(fileName, dataType='dailyTS')
    sdLst.append(data.iloc[0]['datetime'])
    edLst.append(data.iloc[-1]['datetime'])
    tempLst.append(siteNo)
dataDict = dict(siteNo=siteNoLst, sd=sdLst, ed=edLst)
pd.DataFrame(dataDict).to_csv(os.path.join(modelDir, 'siteNoSel'), index=False)

# prepare shapefile
fileSel = os.path.join(modelDir, 'siteNoSel')
outShapeFile = os.path.join(modelDir, 'basinSel.shp')
siteNoLst = pd.read_csv(fileSel, header=None, dtype=str)[0].tolist()
gageII.extractBasins(siteNoLst, outShapeFile)

# gridMetMask.py
# gridMetExtract.py
# gridMetFromRaw.py
