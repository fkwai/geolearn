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
wqData.c = wqData.c * wqData.q[-1, :, 0:1]
saveName = os.path.join(kPath.dirWQ, 'trainData','loadRef')
np.savez(saveName, q=wqData.q, f=wqData.f,
         c=wqData.c, g=wqData.g, cf=wqData.cf)
