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
from sklearn.linear_model import LinearRegression


wqData = waterQuality.DataModelWQ('HBN')
codeLst = ['00955']
trainset = '00955-Y8090'
testset = '00955-Y0010'
siteNoLst = wqData.info['siteNo'].unique().tolist()
infoTrain = wqData.info.iloc[wqData.subset[trainset]]
infoTest = wqData.info.iloc[wqData.subset[testset]]

varTup = (wqData.varF, wqData.varG, wqData.varQ, wqData.varC)
dataTup1, statTup = wqData.transIn(subset=trainset, varTup=varTup)
dataTup2 = wqData.transIn(subset=trainset, varTup=varTup, statTup=statTup)
ic = [wqData.varC.index(code) for code in codeLst]
x1 = dataTup1[0][-1, :, :]
y1 = dataTup1[3][:, ic]
x2 = dataTup2[0][-1, :, :]
y2 = dataTup2[3][:, ic]

siteNo = siteNoLst[0]

ind1 = infoTrain[infoTrain['siteNo'] == siteNo].index
ind2 = infoTest[infoTest['siteNo'] == siteNo].index
xx1 = x1[ind1, :]
yy1 = y1[ind1, :]
xx2 = x2[ind2, :]

model = LinearRegression().fit(xx1, yy1)
yy2=model.predict(xx2)