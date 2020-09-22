from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
import json
import os


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictNB_y16n36.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']

freqLst = ['W', 'D']
optCLst = ['end', 'seq']
for freq in freqLst:
    for optC, optLabel in zip(optCLst, ['', 'T']):
        dataName = 'nb'+freq+optLabel
        rho = 365 if freq == 'D' else 52
        print(dataName)
        wqData = waterQuality.DataModelWQ.new(
            dataName, siteNoLst, rho=rho, freq=freq, optC=optC)
