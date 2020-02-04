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
siteC = usgs.readUsgsText(fileInvC)

codeLst = \
    ['00915', '00925', '00930', '00935', '00955', '00940', '00945']+\
    ['00418','00419','39086','39087']+\
    ['00301','00300','00618','00681','00653']+\
    ['00010','00530','00094']+\
    ['00403','00408']

codeSet1=sorted(set(siteC['parm_cd'].astype(int)))
codeSet=sorted(set(siteC['parm_cd']))


aa=siteC['parm_cd'].astype(str)

aa=siteC.loc[(siteC['parm_cd'].astype(str) == '70')]
aa['site_no'].astype(str)

codeSet=sorted(set(siteC['parm_cd'].astype(str)))
len(codeSet)

dictTab = dict()
for code in codeLst:
    site = siteC.loc[(siteC['parm_cd'] == code) & (siteC['count_nu'] > 1)]
    temp = dict(
        zip(site['site_no'].tolist(),
            site['count_nu'].astype(int).tolist()))
    dictTab[code] = temp
tabC = pd.DataFrame.from_dict(dictTab)
tabC = tabC.rename_axis('site_no').reset_index()


