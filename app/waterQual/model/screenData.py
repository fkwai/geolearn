import matplotlib.pyplot as plt
from hydroDL.data import usgs, gageII, gridMET
from hydroDL import kPath
from hydroDL.app import waterQuality

import pandas as pd
import numpy as np
import os
import time

# list of site - generate from checkCQ.py, 5978 sites in total
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

# load all data - wrap all data takes 2hrs, 5835 sites left
siteNoLst = siteNoLstAll[:10]
caseName = 'temp'
waterQuality.wrapData(caseName, siteNoLst)
t0 = time.time()
dictData, info, q, c, f, g = waterQuality.loadData(caseName)
print(time.time()-t0)

# count sample numbers of sites
dfSite = info['siteNo'].value_counts().rename(
    'count').to_frame().rename_axis(index='siteNo')
# dfGageII = gageII.readData(
#     varLst=['CLASS'], siteNoLst=dfSite.index.tolist())
# dfGageII = gageII.updateCode(dfGageII)
# dfSite = dfSite.join(dfGageII)
plt.hist(dfSite.values, bins=range(0, 1000, 10))
plt.show()

siteNoSel = dfSite.index[dfSite['count'] > 10]
codeLst = dictData['varC']
codeTemp = ['00010', '00095']
[codeLst.index(code) for code in codeTemp]