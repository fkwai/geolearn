"""
explore what concentration variables are available in USGS NWIS
"""

from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os

# read site inventory
workDir = os.path.join(kPath.dirData, 'USGS', 'inventory')
modelDir = os.path.join(workDir, 'modelUsgs2')
fileInvC = os.path.join(workDir, 'inventory_NWIS_sample')
fileInvQ = os.path.join(workDir, 'inventory_NWIS_streamflow')

siteC = usgs.readUsgsText(fileInvC)
siteQ = usgs.readUsgsText(fileInvQ)
tabGageII = gageII.readTab('bas_classif')

# summarize count for all site
fileCountC = os.path.join(workDir, 'count_NWIS_sample_all')
if os.path.exists(fileCountC):
    tabC = pd.read_csv(fileCountC, dtype={'site_no': str})
else:
    codeLst = np.sort(siteC['parm_cd'].unique()).tolist()
    nSite = len(codeLst)
    dictTab = dict()
    t0 = time.time()
    for k, code in enumerate(codeLst):
        site = siteC.loc[(siteC['parm_cd'] == code) & (siteC['count_nu'] > 1)]
        temp = dict(
            zip(site['site_no'].tolist(),
                site['count_nu'].astype(int).tolist()))
        dictTab[code] = temp
        print('\t site {}/{} time cost {:.2f}'.format(
            k, len(codeLst), time.time()-t0), end='\r')
    tabC = pd.DataFrame.from_dict(dictTab)
    tabC = tabC.rename_axis('site_no').reset_index()
    tabC.to_csv(fileCountC, index=False)

# select sites
idLstQ = siteQ['site_no'].tolist()
idLstG = tabGageII['STAID'].tolist()
siteNoLst = list(set(idLstQ).intersection(set(idLstG)))
tabC = tabC.set_index('site_no')
tabC = tabC[tabC.index.isin(siteNoLst)]
tabC = tabC.drop(columns=tabC.columns[tabC.isna().all()])
tabC.to_csv(os.path.join(workDir, 'count_NWIS_sample_gageII'))

# warp up to a table of all variables of - number of samples, sites, sites of samples >10
seriesLst = [tabC.sum().rename('sum'),
             tabC[tabC <= 10].count().rename('<=10'),
             tabC[tabC > 10].count().rename('10-50'),
             tabC[(tabC > 50) & (tabC < 100)].count().rename('50-100'),
             tabC[tabC > 100].count().rename('sites>100')
             ]
dfRef = usgs.readUsgsText(os.path.join(workDir, 'usgs_parameter_code'))
dfCode = pd.concat(seriesLst, axis=1)
dfCode = dfCode.join(dfRef.set_index('parameter_cd'))
dfCode.to_csv(os.path.join(workDir, 'count_NWIS_code_gageII.csv'))
