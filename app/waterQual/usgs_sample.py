# read inventory of all sites
from hydroDL.data import usgs
import pandas as pd
import time
# read site inventory
fileName = r'C:\Users\geofk\work\waterQuality\inventory_NWIS_sample'
siteAll = usgs.readUsgsText(fileName)
# codeLst = ['00608', '00625', '00631', '00665', '80154']
# nS = 50

# organize to code
t0 = time.time()
codeLst = pd.unique(
    siteAll['parm_cd']).astype(str).tolist().sort(reverse=False)
idLst = pd.unique(siteAll['site_no']).tolist()

mat = np.full([len(idLst), len(codeLst)], np.nan)
# for j,id in enumerate(idLst):

j = 0
id = idLst[j]
site = siteAll.loc[siteAll['site_no'] == id]
code = site['parm_cd'].values
count = site['count_nu'].values
C, ind1, ind2 = np.intersect1d(code, codeLst, return_indices=True)
