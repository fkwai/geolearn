
from sklearn.decomposition import PCA
from hydroDL.model import trainBasin
from hydroDL.data import dbBasin, gageII, gridMET
from hydroDL.master import basinFull
import numpy as np
from hydroDL import utils
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from hydroDL.model import waterNetGlobal
import importlib

importlib.reload(waterNetGlobal)

dataName = 'HBN_Q90ref'
# dataName = 'temp'
DF = dbBasin.DataFrameBasin(dataName)

data = DF.f[:, :, DF.varF.index('LAI')]
fig, ax = plt.subplots(1, 1)
ax.hist(data.flatten(), bins=30)
fig.show()
