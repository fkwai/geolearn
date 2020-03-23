from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins

import pandas as pd
import numpy as np
import os
import time

# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist()
             if siteNo in siteNoLstAll]

# wrap up data
if not waterQuality.exist('HBN'):
    wqData = waterQuality.DataModelWQ.new('HBN', siteNoHBN)
if not waterQuality.exist('HBN-30d'):
    wqData = waterQuality.DataModelWQ.new('HBN-30d', siteNoHBN, rho=30)
if not waterQuality.exist('HBN-5s'):
    wqData = waterQuality.DataModelWQ.new('HBN-5s', siteNoHBN[:5])
if not waterQuality.exist('HBN-5s-30d'):
    wqData = waterQuality.DataModelWQ.new('HBN-5s-30d', siteNoHBN[:5], rho=30)

nE = 100
sE = 50

caseLst = list()
saveName = 'HBN-first50-q'
caseName = basins.wrapMaster(
    dataName='HBN', trainName='first50', batchSize=[None, 200],
    outName=saveName, varYC=None, nEpoch=nE, saveEpoch=sE)
caseLst.append(caseName)

caseLst = list()
saveName = 'HBN-first80-q'
caseName = basins.wrapMaster(
    dataName='HBN', trainName='first80', batchSize=[None, 200],
    outName=saveName, varYC=None, nEpoch=nE, saveEpoch=sE)
caseLst.append(caseName)
# basins.trainModelTS(caseName)

codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst[1:]:
    codeLst = codePdf[codePdf.group == group].index.tolist()
    saveName = 'HBN-first50-opt1-'+group
    caseName = basins.wrapMaster(
        dataName='HBN', trainName='first50', batchSize=[None, 200],
        outName=saveName, varYC=codeLst)
    caseLst.append(caseName)

    saveName = 'HBN-first50-opt2-'+group
    caseName = basins.wrapMaster(
        dataName='HBN', trainName='first50', batchSize=[None, 200],
        outName=saveName, varYC=codeLst, varX=usgs.varQ+gridMET.varLst)
    caseLst.append(caseName)
