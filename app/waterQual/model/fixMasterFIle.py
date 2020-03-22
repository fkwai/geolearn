from hydroDL import kPath
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL.data import gridMET, usgs, gageII
import json

import os
import pandas as pd
import numpy as np

modelFolder = os.path.join(kPath.dirWQ, 'model')
outLst = os.listdir(modelFolder)
for out in outLst:
    master = basins.loadMaster(out)
    master['varYC'] = usgs.varC
    master['varXC'] = gageII.lstWaterQuality
    if master['optQ'] == 1:
        master['varX'] = gridMET.varLst
        master['varY'] = usgs.varQ
    if master['optQ'] == 2:
        master['varX'] = usgs.varQ+gridMET.varLst
        master['varY'] = None
    if master['optQ'] == 3:
        master['varX'] = gridMET.varLst
        master['varY'] = None
    if master['optQ'] == 4:
        master['varX'] = usgs.varQ
        master['varY'] = None
    outFolder = os.path.join(modelFolder, out)
    with open(os.path.join(outFolder, 'master.json'), 'w') as fp:
        json.dump(master, fp)
