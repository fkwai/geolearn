"""
found many huge negtive values in streamflow observation. Figure out why
"""
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np

dataName = 'sbW'
wqData = waterQuality.DataModelWQ(dataName)

q = wqData.q
c = wqData.c

[ind1, ind2, ind3] = np.where(q < -0)
len(ind2)
siteNoLst = wqData.info.loc[ind2]['siteNo'].unique()

siteNo = siteNoLst[0]
dfQ = usgs.readStreamflow(siteNo)
dfQ = dfQ.dropna(how='all')
aa = dfQ[dfQ['00060_00003'] <= 0]

# [ind1, ind2] = np.where(c <= -0)

siteNo = '06471000'
usgs.readStreamflow


a=np.less(q, 0., where=~np.isnan(q))
q[a]=0