from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
import json
import os


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
# nsLst = [5, 2]
nsLst = [5]
for ns in nsLst:
    # ns = 5
    dataName = 'dictRB_Y30N{}.json'.format(ns)
    with open(os.path.join(dirSel, dataName)) as f:
        dictSite = json.load(f)
    siteNoLst = dictSite['comb']
    # freqLst = ['W', 'D']
    freqLst = ['W']
    # optCLst = ['seq']
    optC = 'end'
    for freq in freqLst:
        # dataName = 'rbT'+freq+'N'+str(ns)
        dataName = 'rbT'+freq+'N'+str(ns)+'T1'
        # rho = 365 if freq == 'D' else 52
        rho = 365 if freq == 'D' else 1
        print(dataName)
        wqData = waterQuality.DataModelWQ.new(
            dataName, siteNoLst, rho=rho, freq=freq, optC=optC)
