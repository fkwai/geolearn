from hydroDL.data import gageII, usgs
from hydroDL import kPath
import os
import fiona
import time
import json
import pandas as pd
import numpy as np

shapeFile = os.path.join(kPath.dirData, 'USGS', 'basins', 'basinN5.shp')
shapeFileOut = os.path.join(kPath.dirData, 'USGS', 'basins', 'basinN5_new.shp')

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
codeLst = sorted(usgs.newC)

matV = np.zeros([len(siteNoLst), len(codeLst)])
for code in codeLst:
    siteNoTemp = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
    indC = codeLst.index(code)
    matV[indS, indC] = 1
dfT = pd.DataFrame(index=siteNoLst, columns=codeLst, data=matV)

shape = fiona.open(shapeFile)
meta = shape.meta
for code in codeLst:    
    meta['schema']['properties'][code] = 'int'
t0 = time.time()
n = len(shape)
with fiona.open(shapeFileOut, 'w', **meta) as output:
    for k, feat in enumerate(shape):
        print('{:.2f}% {:.2f}'.format(k/n*100, time.time()-t0))
        siteNo = feat['properties']['GAGE_ID']
        print(siteNo)
        for code in codeLst:
            feat['properties'][code] = int(dfT.loc[siteNo][code])
        output.write(feat)

        # code = tabLookup.loc[xx]['code']
        # feat['properties']['code'] = int(code)
        
