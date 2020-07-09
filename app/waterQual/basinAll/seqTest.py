from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('basinRef')

# seq test
# outLst = ['basinRef-Yodd-opt1', 'basinRef-Yodd-opt2',
#           'basinRef-Yeven-opt1', 'basinRef-Yeven-opt2']
outLst = [ 'basinRef-Yeven-opt1']
siteNoLst = wqData.info['siteNo'].unique().tolist()
for outName in outLst:
    basins.testModelSeq(outName, siteNoLst, wqData=wqData, ep=500)
