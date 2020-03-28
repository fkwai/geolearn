from hydroDL.master import basins
from hydroDL.app import waterQuality, relaCQ
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from hydroDL import utils

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('HBN')

figFolder = os.path.join(kPath.dirWQ, 'HBN', 'CQ')

siteNoLst = wqData.info['siteNo'].unique().tolist()
codePdf = usgs.codePdf
pdfArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoLst)
unitConv = 0.3048**3*365*24*60*60/1000**2

info = wqData.info

code = '00955'
siteNo = siteNoLst[1]
indS = info[info['siteNo'] == siteNo].index.values
area = pdfArea.loc[siteNo]['DRAIN_SQKM']
q = wqData.q[-1, indS, 0]/area*unitConv
c = wqData.c[indS, wqData.varC.index(code)]

[q, c], ind = utils.rmNan([q, c])
t = info[info['siteNo'] == siteNo]['date'].values[ind]

plt.plot(q,c,'*')
plt.show()
