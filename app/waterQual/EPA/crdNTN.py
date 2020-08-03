from hydroDL.data import gageII, usgs, gridMET
from hydroDL import kPath, utils
import pandas as pd
import numpy as np
import os

dirNTN = os.path.join(kPath.dirData, 'EPA', 'NTN')

# USGS sites
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
varG = ['LAT_GAGE', 'LNG_GAGE', 'CLASS']
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLstAll)
tabG = gageII.updateCode(tabG)
# due to bugs in excel
tabG2=pd.read_csv(os.path.join(dirNTN, 'crdUSGS-temp.csv'), dtype=str)
tabG['x']=tabG2['x'].values
tabG['y']=tabG2['y'].values
tabG.to_csv(os.path.join(dirNTN, 'crdUSGS.csv'))

# NTN sites
fileSite = os.path.join(dirNTN, 'NTNsites.csv')
tabSite = pd.read_csv(fileSite)
varLst = ['siteid', 'latitude', 'longitude']
tabSite[varLst].to_csv(os.path.join(dirNTN, 'crdNTN.csv'))