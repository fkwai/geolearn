from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import usgs
import pandas as pd
import numpy as np
import os
import time
import importlib

# read a full list of USGS code

codeFile = os.path.join(kPath.dirData, 'USGS', 'inventory', 'usgsCodeFull.txt')
dfC = usgs.readUsgsText(codeFile, )
# codeLst = dfC[dfC['group'] == 'Stable Isotopes']['parm_cd'].tolist()
codeLst = ['82085', '82745', '82082']

codeLstWQ = ['00010', '00095', '00300', '00400', '00405', '00410',
             '00440', '00600', '00605', '00618', '00660', '00665',
             '00681', '00915', '00925', '00930', '00935', '00940',
             '00945', '00950', '00955', '70303', '71846', '80154']
codeLstIso = ['82085', '82745', '82082']
codeLst = codeLstWQ+codeLstIso

# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
siteNoLst = list()
countLst = list()
dirC = os.path.join(kPath.dirData, 'USGS', 'sample', 'csvIso')

t0 = time.time()
for i, siteNo in enumerate(siteNoLstAll):
    dfC, dfCF = usgs.readSample(siteNo, codeLst=codeLst, flag=2, csv=False)
    if dfC is not None:
        dfC.to_csv(os.path.join(dirC, siteNo))
        dfCF.to_csv(os.path.join(dirC, siteNo+'_flag'))
        countLst.append(len(dfC))
        siteNoLst.append(siteNo)
    print('{}/{} {:.2f}'.format(
        i, len(siteNoLstAll), time.time()-t0))
