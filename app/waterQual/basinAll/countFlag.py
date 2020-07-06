from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs
import pandas as pd
import numpy as np
import os
import time


# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

codeLst = usgs.codeLst
startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2019, 12, 31)
flagLst = usgs.dfFlagC['label'].to_list()
dfCountFlag = pd.DataFrame(index=flagLst+['other'], columns=codeLst).fillna(0)
# all sites
for k, siteNo in enumerate(siteNoLstAll):
    print(k, siteNo)
    dfC, dfCF = usgs.readSample(siteNo, codeLst=codeLst,
                                startDate=startDate, flag=True)
    for code in codeLst:
        temp = dfCF[code+'_cd'].value_counts()
        temp['other'] = temp[~temp.index.isin(flagLst)].sum()
        dfCountFlag[code] = dfCountFlag[code].add(temp, fill_value=0)
dfCountFlag.astype(int).to_csv('temp1.csv')

# basinRef
tabSel = gageII.readData(
    varLst=['CLASS'], siteNoLst=siteNoLstAll)
tabSel = gageII.updateCode(tabSel)
siteNoLst = tabSel[tabSel['CLASS'] == 1].index.tolist()
dfCountFlag = pd.DataFrame(index=flagLst+['other'], columns=codeLst).fillna(0)
for k, siteNo in enumerate(siteNoLst):
    print(k, siteNo)
    dfC, dfCF = usgs.readSample(siteNo, codeLst=codeLst,
                                startDate=startDate, flag=True)
    for code in codeLst:
        temp = dfCF[code+'_cd'].value_counts()
        temp['other'] = temp[~temp.index.isin(flagLst)].sum()
        dfCountFlag[code] = dfCountFlag[code].add(temp, fill_value=0)
dfCountFlag.astype(int).to_csv('temp1.csv')


siteNo = '402114105350101'
code = '71846'
a = dfCountFlag[code]
b = dfCF[code+'_cd'].value_counts()
b[~b.index.isin(a.index)].sum()
