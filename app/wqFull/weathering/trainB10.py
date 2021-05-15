from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL.master import basinFull
from hydroDL import kPath
import json
import os
import importlib

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'weathering'
freq = 'D'
# DM = dbBasin.DataModelFull.new(
#     dataName, siteNoLst, sdStr=sd, edStr=ed, freq=freq)
DM = dbBasin.DataModelFull(dataName)
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']

label = 'FPR2QC'
varX = dbBasin.label2var(label.split('2')[0])
varY = codeSel
varXC = gageII.varLst
varYC = None
sd = '1982-01-01'
ed = '2009-12-31'
rho = 365
outName = '{}-{}-t{}-B10'.format(dataName, label, rho)
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             sd=sd, ed=ed, nEpoch=100,
                             batchSize=[rho, 200], nIterEp=20)
basinFull.trainModel(outName)
