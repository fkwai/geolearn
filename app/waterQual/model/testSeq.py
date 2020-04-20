import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

doLst = list()
# doLst.append('data')
# doLst.append('subset')
# doLst.append('train')

if 'data' in doLst:
    # only look at 5 site with most 00955 obs
    # ['11264500', '07083000', '01466500', '04063700', '10343500']
    dataName = 'HBN'
    codeLst = ['00618', '00955']
    wqData = waterQuality.DataModelWQ(dataName)
    icLst = [wqData.varC.index(code) for code in codeLst]
    indAll = np.where(~np.isnan(wqData.c[:, icLst]).all(axis=1))[0]
    siteNoHBN = wqData.info['siteNo'].unique()
    info = wqData.info.iloc[indAll]
    tabCount = info.groupby('siteNo').count()
    siteNoLst = tabCount.nlargest(5, 'date').index.tolist()
    wqData = waterQuality.DataModelWQ.new('HBN5', siteNoLst)

if 'subset' in doLst:
    wqData = waterQuality.DataModelWQ('HBN5')
    codeLst = ['00618', '00955']
    icLst = [wqData.varC.index(code) for code in codeLst]
    indAll = np.where(~np.isnan(wqData.c[:, icLst]).all(axis=1))[0]
    indAny = np.where(~np.isnan(wqData.c[:, icLst]).any(axis=1))[0]
    wqData.saveSubset('-'.join(sorted(codeLst)+['all']), indAll)
    wqData.saveSubset('-'.join(sorted(codeLst)+['any']), indAll)
    for ind, lab in zip([indAll, indAny], ['all', 'any']):
        indYr1 = waterQuality.indYr(
            wqData.info.iloc[ind], yrLst=[1979, 2000])[0]
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y8090']), indYr1)
        indYr2 = waterQuality.indYr(
            wqData.info.iloc[ind], yrLst=[2000, 2020])[0]
        wqData.saveSubset('-'.join(sorted(codeLst)+[lab, 'Y0010']), indYr2)

if 'training' in doLst:
    dataName = 'HBN5'
    codeLst = ['00618', '00955']
    trainset = '00618-00955-all-Y8090'
    testset = '00618-00955-all-Y0010'
    out = 'HBN5-00618-00955-all-Y8090'
    wqData = waterQuality.DataModelWQ(dataName)
    masterName = basins.wrapMaster(
        dataName='HBN5', trainName=trainset, batchSize=[
            None, 100], outName=out, varYC=codeLst, nEpoch=100)
    basins.trainModelTS(masterName)


# sequence testing
dataName = 'HBN'
outName = 'HBN-00618-00955-all-Y8090-opt2'
trainset = '00618-00955-all-Y8090'
testset = '00618-00955-all-Y0010'

wqData = waterQuality.DataModelWQ(dataName)

# point testing
yP, ycP = basins.testModel(outName, testset, wqData=wqData)

# sequence testing
infoTrain = wqData.info.iloc[wqData.subset[trainset]]
siteNoLst = infoTrain['siteNo'].unique().tolist()
dictPred = basins.testModelSeq(outName, siteNoLst=siteNoLst, wqData=wqData)
