
import time
from hydroDL import kPath
import os
import numpy as np
import pandas as pd
import json
from hydroDL.data import gageII

fileSiteNo = os.path.join(kPath.dirUSGS, 'basins', 'siteNoLst.json')
with open(fileSiteNo) as fp:
    dictSite = json.load(fp)
sLst0 = dictSite['CONUS']

outDir=r'F:\USGS\gridMET\output'

sLst1 = [f for f in sorted(os.listdir(outDir))]
sLst2 = [f for f in sLst0 if f not in sLst1]

dfG = gageII.readData(siteNoLst=sLst2)

(dfG['CLASS']=='Ref').values.sum()