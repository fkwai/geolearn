import importlib
from hydroDL.app import waterQuality, wqLinear
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
from sklearn.linear_model import LinearRegression

wqData = waterQuality.DataModelWQ('Silica64')
optX = 'F'
optT = 'Y8090'
code = '00955'
order = (5, 0, 5)
siteNoLst = wqData.siteNoLst

for optT in ['Y8090', 'Y0010']:
    for siteNo in siteNoLst:
        dfP = wqLinear.loadSeq(siteNo, code, 'ARMA',
                               optX=optX, optT=optT, order=(5, 0, 0))
