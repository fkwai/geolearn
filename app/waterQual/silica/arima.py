import importlib
from hydroDL.app import waterQuality
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

siteNo = '01124000'
dfX = waterQuality.readSiteX(siteNo, gridMET.varLst)
dfY = waterQuality.readSiteY(siteNo, ['00955'])


mod = sm.tsa.statespace.SARIMAX(dfY, exog=dfX, order=(100, 0, 0))
res = mod.fit()
print(res.summary())
