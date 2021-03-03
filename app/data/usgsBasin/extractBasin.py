from hydroDL.data import gageII
from hydroDL import kPath
import os
import json

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

outShapeFile = os.path.join(kPath.dirData, 'USGS', 'basins', 'basinN5.shp')
gageII.extractBasins(siteNoLst, outShapeFile)
